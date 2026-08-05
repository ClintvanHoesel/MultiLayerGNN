"""Tests for the periodic boundary condition (PBC) radius graph.

Covers:
* ``morphology_gnn.data.radius_graph_pbc`` (pure PyTorch) against a
  brute-force minimum-image reference.
* The CUDA kernel
  (``morphology_gnn.cuda_radius_graph.radius_graph_pbc``) against the
  Python implementation (skipped when CUDA is unavailable).
* Edge cases: ``loop``, ``max_num_neighbors``, empty/single atoms,
  periodic-image folding, lattice input shapes, and the real HDF5 data.
"""

from __future__ import annotations

import glob
import os

import pytest
import torch
from torch_geometric.nn import radius_graph as pyg_radius_graph

from morphology_gnn.cuda_radius_graph import radius_graph_pbc as cuda_radius_graph_pbc
from morphology_gnn.data import H5MolecularDataset, radius_graph_pbc

try:
    import torch as _torch

    _CUDA_AVAILABLE = _torch.cuda.is_available() and cuda_radius_graph_pbc is not None
except Exception:  # pragma: no cover - defensive
    _CUDA_AVAILABLE = False

requires_cuda = pytest.mark.skipif(
    not _CUDA_AVAILABLE, reason="CUDA (or the CUDA extension) is not available"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REAL_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*_ams.hdf5")))
requires_data = pytest.mark.skipif(
    not REAL_FILES, reason="no *_ams.hdf5 files found in data/"
)


# --------------------------------------------------------------------------- #
# Reference implementation (brute-force minimum image, wrapped into the cell)
# --------------------------------------------------------------------------- #
def brute_force_pbc(pos, r, lattice, loop=False, max_num_neighbors=None):
    """Exact PBC radius graph by minimising over all periodic images.

    Positions are first wrapped into the unit cell (as the implementation
    does) so a small shift range suffices. Uses float64 for the reference.
    """
    N = pos.shape[0]
    lattice64 = lattice.to(torch.float64)
    if lattice64.ndim == 1:
        lattice64 = torch.diag(lattice64)
    p = pos.to(torch.float64)
    frac = p @ torch.linalg.inv(lattice64)
    p = (frac - torch.floor(frac)) @ lattice64

    s = torch.arange(-2, 3, dtype=torch.float64)
    shifts = torch.stack(torch.meshgrid(s, s, s, indexing="ij"), dim=-1).reshape(-1, 3)
    images = shifts @ lattice64  # (S, 3)

    edges = []
    for i in range(N):
        d = (p[i] - p).reshape(1, N, 3)
        dist = (d - images.reshape(-1, 1, 3)).norm(dim=-1).min(dim=0).values
        if not loop:
            dist[i] = float("inf")
        valid = (dist <= r).nonzero().flatten()
        if valid.numel() == 0:
            continue
        if max_num_neighbors is not None and max_num_neighbors < valid.numel():
            valid = valid[torch.topk(-dist[valid], k=max_num_neighbors).indices]
        edges.extend((i, int(j)) for j in valid.tolist())

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).T


def undirected_set(edge_index):
    """Canonical set of undirected edges (sort each edge, dedupe)."""
    return torch.unique(torch.sort(edge_index, dim=0).values, dim=1)


def random_config(seed, max_num_neighbors_candidates=(None, 3, 8)):
    """Deterministic random (pos, r, lattice, loop, mnn) configuration."""
    g = torch.Generator().manual_seed(seed)
    N = int(torch.randint(4, 70, (1,), generator=g).item())
    box = torch.randint(7, 22, (3,), generator=g).to(torch.float32)
    r = float((torch.rand(1, generator=g).item() * (box.min().item() / 2 - 0.6)) + 0.4)
    # positions may be outside the box / negative
    pos = torch.rand(N, 3, generator=g) * (box.max().item() + 5) - 2.5
    pos = pos.to(torch.float32)
    loop = bool(torch.randint(0, 2, (1,), generator=g).item())
    mnn = max_num_neighbors_candidates[
        int(torch.randint(0, len(max_num_neighbors_candidates), (1,), generator=g).item())
    ]
    return pos, r, box, loop, mnn


ORTHO_SEEDS = list(range(40))
NONORTHO_LATTICES = [
    torch.tensor([[10.0, 2.0, 1.0], [0.0, 11.0, 1.5], [0.0, 0.0, 12.0]]),
    torch.tensor([[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [0.0, 0.0, 11.0]]),
    torch.tensor([[9.0, 1.0, 0.5], [0.0, 10.0, 1.0], [0.0, 0.0, 11.0]]),
]


# --------------------------------------------------------------------------- #
# Python implementation vs brute-force reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", ORTHO_SEEDS)
def test_orthorhombic_matches_bruteforce(seed):
    pos, r, box, loop, mnn = random_config(seed)
    expected = brute_force_pbc(pos, r, box, loop=loop, max_num_neighbors=mnn)
    got = radius_graph_pbc(pos, r, box, loop=loop, max_num_neighbors=mnn)
    assert torch.equal(undirected_set(expected), undirected_set(got))


@pytest.mark.parametrize("seed", ORTHO_SEEDS)
def test_orthorhombic_3x3_matches_1d(seed):
    """A diagonal 3x3 lattice matrix must give the same result as a 1-D box."""
    pos, r, box, loop, mnn = random_config(seed)
    as_3x3 = torch.diag(box)
    e1 = radius_graph_pbc(pos, r, box, loop=loop, max_num_neighbors=mnn)
    e3 = radius_graph_pbc(pos, r, as_3x3, loop=loop, max_num_neighbors=mnn)
    assert torch.equal(undirected_set(e1), undirected_set(e3))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("lattice_idx", list(range(len(NONORTHO_LATTICES))))
def test_nonorthorhombic_matches_bruteforce(seed, lattice_idx):
    lattice = NONORTHO_LATTICES[lattice_idx]
    g = torch.Generator().manual_seed(1000 + seed)
    N = int(torch.randint(5, 40, (1,), generator=g).item())
    pos = (torch.rand(N, 3, generator=g) * 4.0).to(torch.float32)
    r = float((torch.rand(1, generator=g).item() * 1.5) + 1.0)
    loop = bool(torch.randint(0, 2, (1,), generator=g).item())
    mnn = [None, 4][int(torch.randint(0, 2, (1,), generator=g).item())]

    expected = brute_force_pbc(pos, r, lattice, loop=loop, max_num_neighbors=mnn)
    got = radius_graph_pbc(pos, r, lattice, loop=loop, max_num_neighbors=mnn)
    assert torch.equal(undirected_set(expected), undirected_set(got))


def test_hand_computed_orthorhombic():
    """Hand-computed minimum-image example in a 10 Angstrom box, r = 2."""
    box = torch.tensor([10.0, 10.0, 10.0])
    pos = torch.tensor(
        [
            [1.0, 1.0, 1.0],  # 0
            [9.0, 1.0, 1.0],  # 1 -> min-image distance to 0 is 2
            [5.0, 5.0, 5.0],  # 2 -> isolated
            [1.5, 1.0, 1.0],  # 3 -> distance to 0 is 0.5, to 1 is 2.5 (> r)
        ]
    )
    edge_index = radius_graph_pbc(pos, r=2.0, lattice=box, loop=False)
    expected = torch.tensor([[0, 1, 0, 3], [1, 0, 3, 0]])
    assert torch.equal(
        torch.sort(edge_index, dim=1).values,
        torch.sort(expected, dim=1).values,
    )


def test_pbc_catches_periodic_images():
    """Two atoms straddling the boundary must be connected through PBC."""
    box = torch.tensor([10.0, 10.0, 10.0])
    pos = torch.tensor([[0.1, 5.0, 5.0], [9.9, 5.0, 5.0]])
    # Plain (non-PBC) radius graph: |0.1 - 9.9| = 9.8 > r -> no edge.
    plain = pyg_radius_graph(pos, r=1.0, loop=False)
    assert plain.shape[1] == 0
    # PBC minimum image: |0.1 - 9.9 - 10| = 0.2 <= r -> edge.
    pbc = radius_graph_pbc(pos, r=1.0, lattice=box, loop=False)
    assert undirected_set(pbc).shape[1] == 1
    assert set(map(tuple, undirected_set(pbc).t().tolist())) == {(0, 1)}


def test_matches_pyg_without_pbc_for_isolated_cluster():
    """When no periodic image is within r, PBC must equal plain radius_graph."""
    g = torch.Generator().manual_seed(7)
    box = torch.tensor([50.0, 50.0, 50.0])
    pos = torch.rand(30, 3, generator=g) * 10.0 + 20.0  # cluster near the center
    r = 3.0
    pbc = radius_graph_pbc(pos, r, box, loop=False)
    plain = pyg_radius_graph(pos, r=r, loop=False)
    assert torch.equal(undirected_set(pbc), undirected_set(plain))


def test_loop_includes_self_loops():
    pos = torch.rand(20, 3) * 9.0
    box = torch.tensor([10.0, 10.0, 10.0])
    no_loop = radius_graph_pbc(pos, 3.0, box, loop=False)
    with_loop = radius_graph_pbc(pos, 3.0, box, loop=True)
    # every node has a self-loop when loop=True
    self_loops = torch.stack([torch.arange(20), torch.arange(20)])
    assert set(map(tuple, self_loops.t().tolist())).issubset(
        set(map(tuple, undirected_set(with_loop).t().tolist()))
    )
    # loop=False has no self loops and otherwise equals loop=True minus self loops
    assert not set(map(tuple, self_loops.t().tolist())) & set(
        map(tuple, undirected_set(no_loop).t().tolist())
    )
    without_self = with_loop[:, (with_loop[0] != with_loop[1])]
    assert torch.equal(undirected_set(no_loop), undirected_set(without_self))


def test_max_num_neighbors_respected():
    """Each node has at most max_num_neighbors outgoing edges, equal to ref."""
    pos, r, box, _, _ = random_config(seed=99, max_num_neighbors_candidates=(8,))
    k = 8
    edge_index = radius_graph_pbc(pos, r, box, loop=False, max_num_neighbors=k)
    assert edge_index.shape[0] == 2
    if edge_index.shape[1] > 0:
        _, counts = torch.unique(edge_index[0], return_counts=True)
        assert int(counts.max()) <= k
    expected = brute_force_pbc(pos, r, box, loop=False, max_num_neighbors=k)
    assert torch.equal(undirected_set(expected), undirected_set(edge_index))


def test_empty_and_single_atom():
    box = torch.tensor([10.0, 10.0, 10.0])
    # single atom, no loop -> no edges
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    e = radius_graph_pbc(pos, 3.0, box, loop=False)
    assert e.shape == (2, 0)
    # single atom with loop -> one self edge
    e = radius_graph_pbc(pos, 3.0, box, loop=True)
    assert e.shape == (2, 1) and e[0, 0].item() == 0 and e[1, 0].item() == 0
    # empty input
    e = radius_graph_pbc(torch.empty((0, 3)), 3.0, box, loop=False)
    assert e.shape == (2, 0)


def test_shape_dtype_and_bounds():
    pos, r, box, loop, _ = random_config(seed=5)
    e = radius_graph_pbc(pos, r, box, loop=loop)
    assert e.dtype == torch.long
    assert e.shape[0] == 2
    if e.shape[1] > 0:
        assert e.min().item() >= 0
        assert e.max().item() < pos.shape[0]


def test_deterministic():
    pos, r, box, loop, mnn = random_config(seed=11)
    e1 = radius_graph_pbc(pos, r, box, loop=loop, max_num_neighbors=mnn)
    e2 = radius_graph_pbc(pos, r, box, loop=loop, max_num_neighbors=mnn)
    assert torch.equal(e1, e2)


def test_symmetry_without_max_num_neighbors():
    """Without max_num_neighbors the graph is undirected (both directions)."""
    pos, r, box, _, _ = random_config(seed=13, max_num_neighbors_candidates=(None,))
    e = radius_graph_pbc(pos, r, box, loop=False)
    if e.shape[1] == 0:
        pytest.skip("no edges in this configuration")
    rev = e[[1, 0]]
    assert torch.equal(torch.sort(e, dim=1).values, torch.sort(rev, dim=1).values)


def test_invalid_lattice_shapes_raise():
    pos = torch.rand(10, 3)
    with pytest.raises(ValueError):
        radius_graph_pbc(pos, 2.0, torch.tensor([1.0, 2.0]))  # wrong 1-D size
    with pytest.raises(ValueError):
        radius_graph_pbc(pos, 2.0, torch.zeros(4, 4))  # wrong matrix size


# --------------------------------------------------------------------------- #
# CUDA implementation vs Python implementation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", ORTHO_SEEDS)
@requires_cuda
def test_cuda_matches_python(seed):
    pos, r, box, loop, mnn = random_config(seed)
    py = radius_graph_pbc(pos, r, box, loop=loop, max_num_neighbors=mnn)
    cu = cuda_radius_graph_pbc(pos, r, box, loop=loop, max_num_neighbors=mnn)
    assert torch.equal(undirected_set(py), undirected_set(cu))


@requires_cuda
def test_cuda_matches_python_loop_and_mnn():
    """CUDA must honour loop and max_num_neighbors (regression test)."""
    g = torch.Generator().manual_seed(1234)
    for _ in range(10):
        N = int(torch.randint(10, 60, (1,), generator=g).item())
        box = torch.randint(8, 16, (3,), generator=g).to(torch.float32)
        r = float(
            (torch.rand(1, generator=g).item() * (box.min().item() / 2 - 0.5)) + 0.5
        )
        pos = (torch.rand(N, 3, generator=g) * box.max().item()).to(torch.float32)
        for loop in (False, True):
            for mnn in (None, 4, 10):
                py = radius_graph_pbc(pos, r, box, loop=loop, max_num_neighbors=mnn)
                cu = cuda_radius_graph_pbc(
                    pos, r, box, loop=loop, max_num_neighbors=mnn
                )
                assert torch.equal(
                    undirected_set(py), undirected_set(cu)
                ), f"loop={loop} mnn={mnn} N={N} r={r:.3f}"


@requires_cuda
def test_cuda_rejects_nonorthorhombic():
    pos = torch.rand(20, 3)
    tri = torch.tensor([[10.0, 2.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]])
    with pytest.raises(ValueError):
        cuda_radius_graph_pbc(pos, 2.0, tri, loop=False)


# --------------------------------------------------------------------------- #
# Real HDF5 data
# --------------------------------------------------------------------------- #
@requires_data
def test_h5_dataset_exposes_frames():
    """The H5 dataset must expose each MD frame as a valid PyG graph."""
    ds = H5MolecularDataset(REAL_FILES[0], "Positive VIP", radius=6.0)
    assert len(ds) > 0
    data = ds[0]
    assert data.pos.ndim == 2 and data.pos.shape[1] == 3
    assert data.edge_index.dtype == torch.long
    assert data.edge_index.shape[0] == 2
    if data.edge_index.shape[1] > 0:
        assert int(data.edge_index.max()) < data.pos.shape[0]
    assert data.y.numel() == 1


@requires_data
@requires_cuda
def test_cuda_matches_python_on_real_data():
    """CUDA and Python must agree on several real molecule frames."""
    ds = H5MolecularDataset(REAL_FILES[0], "Positive VIP", radius=6.0)
    for idx in range(min(3, len(ds))):
        data = ds[idx]
        py = radius_graph_pbc(data.pos, 6.0, data.lattice, loop=False)
        cu = cuda_radius_graph_pbc(data.pos, 6.0, data.lattice, loop=False)
        assert torch.equal(undirected_set(py), undirected_set(cu))
        # also verify against the exact reference
        ref = brute_force_pbc(data.pos, 6.0, data.lattice, loop=False)
        assert torch.equal(undirected_set(py), undirected_set(ref))
