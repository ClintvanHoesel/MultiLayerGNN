"""Tests for the diffusion position-generator.

Covers:
* ``NoiseSchedule`` endpoints / monotonicity.
* ``min_image_disp`` against a brute-force minimum-image reference.
* ``rebuild_pbc_edges`` (batched) against per-graph ``radius_graph_pbc``.
* ``DiffusionMoleculeModel`` forward shapes + SE(3) equivariance.
* A small capacity/trivial-fit check (the denoiser can fit a fixed pair).
* ``DiffusionMoleculeModule._corrupt`` keeps noisy coordinates in the cell.
* DDPM/DDIM samplers return finite in-cell coordinates; DDIM is deterministic.
* The real HDF5 data exposes the ``box`` attribute that batches to ``(B, 3)``.
"""

from __future__ import annotations

import glob
import os

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from morphology_gnn.data import H5MolecularDataset
from morphology_gnn.model.diffusion_model import (
    DiffusionMoleculeModel,
    NoiseSchedule,
    min_image_disp_batched,
)
from morphology_gnn.model.diffusion_trainer import (
    DiffusionMoleculeModule,
    min_pair_dist,
)
from morphology_gnn.radius_graph import (
    min_image_disp,
    radius_graph_pbc,
    rebuild_pbc_edges,
    wrap_pos,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REAL_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*_ams.hdf5")))
requires_data = pytest.mark.skipif(
    not REAL_FILES, reason="no *_ams.hdf5 files found in data/"
)


def _small_model(**kwargs):
    """A small diffusion model fast enough for CPU tests."""
    defaults = dict(hidden_dim=16, num_layers=2, num_rbf=8, cell_embed_dim=4)
    defaults.update(kwargs)
    return DiffusionMoleculeModel(**defaults)


def _toy_batch(boxes=(12.0, 10.0, 11.0, 14.0, 13.0, 12.0)):
    """Two small graphs in orthorhombic cells; returns (Batch, box)."""
    box = torch.tensor(boxes).reshape(2, 3)
    n1, n2 = 20, 25
    data = [
        Data(
            x=torch.randint(1, 7, (n1, 1)),
            pos=torch.rand(n1, 3) * box[0],
            y=torch.zeros(1),
            box=box[0].reshape(1, 3),
        ),
        Data(
            x=torch.randint(1, 7, (n2, 1)),
            pos=torch.rand(n2, 3) * box[1],
            y=torch.zeros(1),
            box=box[1].reshape(1, 3),
        ),
    ]
    return Batch.from_data_list(data), box


# --------------------------------------------------------------------------- #
# Noise schedule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["cosine", "linear"])
def test_noise_schedule_endpoints_and_monotonic(kind):
    s = NoiseSchedule(kind)
    t = torch.linspace(0.0, 1.0, 21)
    ab = s.alpha_bar(t)
    sig = s.sigma(t)
    assert torch.allclose(sig, torch.sqrt((1.0 - ab).clamp_min(0.0)), atol=1e-6)
    assert ab[0] == pytest.approx(1.0, abs=1e-6)
    assert sig[0] == pytest.approx(0.0, abs=1e-6)
    assert sig[-1] == pytest.approx(1.0, abs=1e-6)
    assert (sig[1:] >= sig[:-1] - 1e-6).all()  # monotone non-decreasing
    assert (ab[1:] <= ab[:-1] + 1e-6).all()  # monotone non-increasing


def test_noise_schedule_unknown_kind():
    with pytest.raises(ValueError):
        NoiseSchedule("nope")


# --------------------------------------------------------------------------- #
# PBC helpers
# --------------------------------------------------------------------------- #
def brute_force_min_image_disp(pos, edge_index, box):
    """Reference: minimize over the 27 periodic shifts (orthorhombic)."""
    src, dst = edge_index[0], edge_index[1]
    disp = pos[dst] - pos[src]
    best = disp.clone()
    best_norm = best.norm(dim=-1)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                shift = torch.tensor([i * box[0], j * box[1], k * box[2]], dtype=pos.dtype)
                cand = disp - shift
                cnorm = cand.norm(dim=-1)
                better = cnorm < best_norm
                best = torch.where(better.unsqueeze(-1), cand, best)
                best_norm = torch.where(better, cnorm, best_norm)
    return best


def test_min_image_disp_matches_brute_force():
    torch.manual_seed(0)
    N = 30
    box = torch.tensor([10.0, 9.0, 8.0])
    pos = torch.rand(N, 3) * box  # inside the cell
    edge_index = torch.randint(0, N, (2, 50))
    disp = min_image_disp(pos, edge_index, box)
    ref = brute_force_min_image_disp(pos, edge_index, box)
    assert disp.shape == (50, 3)
    assert torch.allclose(disp, ref, atol=1e-5)


def test_min_image_disp_accepts_lattice_matrix():
    torch.manual_seed(0)
    box = torch.tensor([10.0, 9.0, 8.0])
    pos = torch.rand(12, 3) * box
    edge_index = torch.randint(0, 12, (2, 20))
    d_box = min_image_disp(pos, edge_index, box)
    d_mat = min_image_disp(pos, edge_index, torch.diag(box))
    assert torch.allclose(d_box, d_mat, atol=1e-6)


def test_min_image_disp_batched_matches_single():
    torch.manual_seed(0)
    box = torch.tensor([10.0, 9.0, 8.0])
    pos = torch.rand(15, 3) * box
    edge_index = torch.randint(0, 15, (2, 24))
    box_per_node = box.unsqueeze(0).expand(15, 3)
    d_b = min_image_disp_batched(pos, edge_index, box_per_node)
    d_s = min_image_disp(pos, edge_index, box)
    assert torch.allclose(d_b, d_s, atol=1e-6)


def test_wrap_pos():
    pos = torch.tensor([[12.5, -1.0, 20.0]])
    box = torch.tensor([10.0, 10.0, 10.0])
    wrapped = wrap_pos(pos, box)
    assert torch.allclose(wrapped, torch.tensor([[2.5, 9.0, 0.0]]), atol=1e-6)
    assert (wrapped >= 0).all() and (wrapped < box).all()


def test_rebuild_pbc_edges_matches_per_graph():
    torch.manual_seed(0)
    box = torch.tensor([[12.0, 10.0, 11.0], [14.0, 13.0, 12.0]])
    n1, n2 = 20, 25
    pos = torch.cat([torch.rand(n1, 3) * box[0], torch.rand(n2, 3) * box[1]])
    batch = torch.cat([torch.zeros(n1), torch.ones(n2)]).long()
    ei = rebuild_pbc_edges(pos, batch, box, 4.0)
    e0 = radius_graph_pbc(pos[:n1], r=4.0, lattice=box[0], loop=False)
    e1 = radius_graph_pbc(pos[n1:], r=4.0, lattice=box[1], loop=False) + n1
    ref = torch.cat([e0, e1], dim=1)
    assert torch.equal(ei, ref)
    # works with (B, 3, 3) lattice matrices too
    ei_mat = rebuild_pbc_edges(pos, batch, torch.stack([torch.diag(b) for b in box]), 4.0)
    assert torch.equal(ei_mat, ref)


# --------------------------------------------------------------------------- #
# Model forward + equivariance
# --------------------------------------------------------------------------- #
def test_forward_shapes():
    torch.manual_seed(0)
    model = _small_model()
    box = torch.tensor([[12.0, 10.0, 11.0], [14.0, 13.0, 12.0]])
    n1, n2 = 10, 14
    x = torch.cat([torch.randint(1, 7, (n1,)), torch.randint(1, 7, (n2,))])
    pos = torch.cat([torch.rand(n1, 3) * box[0], torch.rand(n2, 3) * box[1]])
    batch = torch.cat([torch.zeros(n1), torch.ones(n2)]).long()
    t = torch.tensor([0.3, 0.7])
    ei = rebuild_pbc_edges(pos, batch, box, 4.0)
    out = model(x, pos, ei, batch, t, box)
    assert out.shape == (n1 + n2, 3)
    assert torch.isfinite(out).all()
    # atom types stored as (N, 1) (as PyG datasets do) must also work
    out2 = model(x.unsqueeze(-1), pos, ei, batch, t, box)
    assert out2.shape == (n1 + n2, 3)


def _random_rotation():
    A = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q = Q.clone()
        Q[:, 0] *= -1
    return Q


def test_se3_equivariance():
    """Rotation is covariant, translation invariant (large-cell / no-wrap path).

    Uses a large box with the structure centered in it so periodic wrapping is
    the identity and the minimum-image displacements reduce to the naive
    differences — the regime where the continuous-space SE(3) property holds.
    """
    torch.manual_seed(0)
    model = _small_model(dropout=0.0).eval()
    N = 15
    box = torch.tensor([30.0, 30.0, 30.0])
    center = box / 2
    pos = (torch.rand(N, 3) * 6 - 3) + center  # centered, small spread
    x = torch.randint(1, 7, (N,))
    batch = torch.zeros(N, dtype=torch.long)
    t = torch.tensor([0.4])
    box_b = box.unsqueeze(0)

    edge_index = radius_graph_pbc(pos, r=4.0, lattice=box, loop=False)
    eps = model(x, pos, edge_index, batch, t, box_b)  # (N, 3)

    # Rotation about the box center.
    R = _random_rotation()
    pos_rot = (pos - center) @ R.t() + center
    edge_rot = radius_graph_pbc(pos_rot, r=4.0, lattice=box, loop=False)
    eps_rot = model(x, pos_rot, edge_rot, batch, t, box_b)
    assert torch.allclose(eps_rot, eps @ R.t(), atol=1e-3, rtol=1e-3)

    # Small translation (stays in the cell -> wrapping is the identity).
    delta = torch.tensor([0.3, -0.2, 0.5])
    pos_tr = pos + delta
    edge_tr = radius_graph_pbc(pos_tr, r=4.0, lattice=box, loop=False)
    eps_tr = model(x, pos_tr, edge_tr, batch, t, box_b)
    assert torch.allclose(eps_tr, eps, atol=1e-3, rtol=1e-3)


def test_trivial_fit():
    """Capacity check: a small denoiser fits a fixed (noisy, target) pair."""
    torch.manual_seed(0)
    model = _small_model(dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    N = 30
    box = torch.tensor([20.0, 20.0, 20.0])
    pos0 = torch.rand(N, 3) * 10
    x = torch.randint(1, 7, (N,))
    batch = torch.zeros(N, dtype=torch.long)
    box_b = box.unsqueeze(0)
    t = torch.full((1,), 0.5)
    eps_target = torch.randn(N, 3) * 0.5
    sigma = model.noise_schedule.sigma(t)
    x_noisy = pos0 + sigma[batch].unsqueeze(-1) * eps_target
    edge_index = radius_graph_pbc(x_noisy, r=4.0, lattice=box, loop=False)

    loss0 = None
    for _ in range(500):
        opt.zero_grad()
        eps_hat = model(x, x_noisy, edge_index, batch, t, box_b)
        loss = F.mse_loss(eps_hat, eps_target)
        loss.backward()
        opt.step()
        if loss0 is None:
            loss0 = loss.item()
    assert loss.item() < 0.3 * loss0  # clearly improved


# --------------------------------------------------------------------------- #
# Lightning module
# --------------------------------------------------------------------------- #
def test_corrupt_in_cell():
    torch.manual_seed(0)
    module = DiffusionMoleculeModule(_small_model(), radius=4.0)
    b, box = _toy_batch()
    assert b.box.shape == (2, 3)
    x_noisy, edge_index, t, eps = module._corrupt(b)
    assert x_noisy.shape == b.pos.shape
    assert edge_index.shape[0] == 2
    assert t.shape == (2,)
    assert eps.shape == b.pos.shape
    box_node = module._batch_box(b)[b.batch]
    assert (x_noisy >= -1e-6).all() and (x_noisy < box_node + 1e-6).all()


def test_batch_box_robust_to_3vec_collation():
    """A (3,) box stored per graph collates to (B*3,); the helper reshapes."""
    n = 8
    data = [
        Data(x=torch.randint(1, 7, (n, 1)), pos=torch.rand(n, 3), y=torch.zeros(1), box=torch.tensor([10.0, 10.0, 10.0])),
        Data(x=torch.randint(1, 7, (n, 1)), pos=torch.rand(n, 3), y=torch.zeros(1), box=torch.tensor([12.0, 12.0, 12.0])),
    ]
    b = Batch.from_data_list(data)
    module = DiffusionMoleculeModule(_small_model(), radius=4.0)
    assert module._batch_box(b).shape == (2, 3)


def test_sample_finite_in_cell_and_ddim_deterministic():
    torch.manual_seed(0)
    module = DiffusionMoleculeModule(_small_model(), radius=4.0, sample_steps=10)
    N = 12
    atoms = torch.randint(1, 7, (N,))
    box = torch.tensor([15.0, 12.0, 13.0])

    for ddim in (False, True):
        gen = module.sample(atoms, box, steps=10, ddim=ddim, seed=0)
        assert gen.shape == (N, 3)
        assert torch.isfinite(gen).all()
        assert (gen >= 0).all() and (gen < box).all()
        assert torch.isfinite(min_pair_dist(gen))
        assert module.training  # module returned to train mode after sampling

    g1 = module.sample(atoms, box, steps=10, ddim=True, seed=1)
    g2 = module.sample(atoms, box, steps=10, ddim=True, seed=1)
    assert torch.allclose(g1, g2, atol=1e-6)


def test_sample_many_shapes():
    torch.manual_seed(0)
    module = DiffusionMoleculeModule(_small_model(), radius=4.0, sample_steps=4)
    atoms = torch.randint(1, 7, (10,))
    box = torch.tensor([14.0, 12.0, 11.0])
    out = module.sample_many(atoms, box, n=3, seed=5)
    assert out.shape == (3, 10, 3)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #
@requires_data
def test_real_data_has_box():
    ds = H5MolecularDataset(REAL_FILES[0], "Positive VIP", radius=4.0)
    d = ds[0]
    assert hasattr(d, "box")
    assert d.box.shape == (1, 3)
    assert hasattr(d, "pos") and d.pos.shape[1] == 3
    b = Batch.from_data_list([ds[0], ds[1]])
    assert b.box.shape == (2, 3)
