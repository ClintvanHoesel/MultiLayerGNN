"""Tests for the sequential (+z) point-by-point sampler and z-ordered training.

Covers:
* ``DiffusionMoleculeModule.sample_sequential_z``:
  - no generated molecule enters the excluded bottom layer;
  - molecules are produced sequentially in +z (each at/above the current
    z-frontier; with ``z_step`` they are strictly separated);
  - an ``initial_pos`` structure seeds the context and raises the frontier;
  - batched / independent structures each advance their own frontier;
  - edge cases: zero / one point, more points than species, negative count;
  - reproducibility (a seeded run is fully deterministic, also on GPU);
  - the model actually receives the growing structure as context.
* ``_corrupt_z_ordered`` (z-ordered training): it reads the dataset-provided
  per-node ``target_mask`` (the studied molecule; everything below it is clean
  context), only that target is noised, and the training loss is computed on it
  — matching the sampler.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from morphology_gnn.model.diffusion_model import DiffusionMoleculeModel
from morphology_gnn.model.diffusion_trainer import DiffusionMoleculeModule


def _small_model(**kwargs):
    defaults = dict(
        hidden_dim=16, num_layers=2, num_rbf=8, cell_embed_dim=4, dropout=0.0
    )
    defaults.update(kwargs)
    return DiffusionMoleculeModel(**defaults)


def _small_module(z_ordered: bool = False, **kwargs) -> DiffusionMoleculeModule:
    model = _small_model()
    return DiffusionMoleculeModule(
        model, radius=6.0, sample_steps=8, z_ordered=z_ordered, **kwargs
    )


def _toy_batch():
    """Two small graphs whose highest-z molecule is the marked target.

    Mirrors the z-ordered box dataset: the studied (target) molecule is the
    highest-z node and everything below it is clean context.
    """
    box = torch.tensor([[20.0, 20.0, 40.0], [22.0, 22.0, 40.0]])
    data = [
        Data(
            x=torch.randint(0, 3, (6, 1)),
            pos=torch.tensor(
                [[1, 1, z] for z in [5, 8, 20, 35, 37, 39]], dtype=torch.float
            ),
            y=torch.zeros(1),
            box=box[0].reshape(1, 3),
            target_mask=torch.tensor([False] * 5 + [True]),
        ),
        Data(
            x=torch.randint(0, 3, (5, 1)),
            pos=torch.tensor(
                [[2, 2, z] for z in [3, 10, 12, 30, 38]], dtype=torch.float
            ),
            y=torch.zeros(1),
            box=box[1].reshape(1, 3),
            target_mask=torch.tensor([False] * 4 + [True]),
        ),
    ]
    return Batch.from_data_list(data), box


# --------------------------------------------------------------------------- #
# Sequential sampler: constraints
# --------------------------------------------------------------------------- #
def test_sequential_z_no_point_below_bottom_exclusion():
    mod = _small_module()
    species = torch.randint(0, 3, (8,))
    box = torch.tensor([20.0, 20.0, 100.0])
    bottom = 12.0
    gen = mod.sample_sequential_z(
        species, box, num_points=8, bottom_z_exclusion=bottom, seed=0
    )
    assert gen.shape == (8, 3)
    assert torch.isfinite(gen).all()
    # The excluded bottom layer never contains a newly generated molecule.
    assert (gen[:, 2] >= bottom - 1e-4).all()
    # Positions stay inside the cell.
    assert (gen >= -1e-4).all() and (gen <= box + 1e-4).all()


def test_sequential_z_points_at_or_above_current_frontier():
    """Each new point is at/above the max z of everything placed before it."""
    mod = _small_module()
    species = torch.randint(0, 3, (8,))
    box = torch.tensor([20.0, 20.0, 100.0])
    bottom = 5.0
    gen = mod.sample_sequential_z(
        species, box, num_points=8, bottom_z_exclusion=bottom, seed=0
    )
    z = gen[:, 2]
    for i in range(z.shape[0]):
        frontier = max(bottom, float(z[:i].max()) if i > 0 else float("-inf"))
        assert z[i] >= frontier - 1e-4, f"point {i} below its frontier {frontier}"
    # hence non-decreasing in z
    assert (z[1:] >= z[:-1] - 1e-4).all()


def test_sequential_z_z_step_minimum_separation():
    """With z_step, consecutive molecules are strictly separated in z."""
    mod = _small_module()
    species = torch.randint(0, 3, (6,))
    # Tall cell: the frontier never runs out of room, so the separation is a
    # hard guarantee regardless of where the (random) model places the points.
    box = torch.tensor([20.0, 20.0, 500.0])
    step = 3.0
    gen = mod.sample_sequential_z(
        species, box, num_points=6, bottom_z_exclusion=10.0, z_step=step, seed=0
    )
    z = gen[:, 2]
    assert z[0] >= 10.0 - 1e-4
    assert (z[1:] >= z[:-1] + step - 1e-4).all()


# --------------------------------------------------------------------------- #
# Sequential sampler: initial structure / batching / edge cases
# --------------------------------------------------------------------------- #
def test_sequential_z_initial_structure_sets_frontier():
    mod = _small_module()
    species = torch.randint(0, 3, (4,))
    box = torch.tensor([20.0, 20.0, 100.0])
    init_types = torch.tensor([0, 1])

    # Initial structure extends above the bottom boundary -> the frontier
    # follows the initial max z (33), not the bottom (5).
    init_high = torch.tensor([[1.0, 1.0, 30.0], [2.0, 2.0, 33.0]])
    gen = mod.sample_sequential_z(
        species,
        box,
        num_points=4,
        bottom_z_exclusion=5.0,
        initial_pos=init_high,
        initial_types=init_types,
        seed=1,
    )
    assert (gen[:, 2] >= 33.0 - 1e-4).all()

    # Initial structure entirely below the allowed region -> the frontier stays
    # at the bottom exclusion (10), so the generated molecules are above it.
    init_low = torch.tensor([[1.0, 1.0, 2.0], [2.0, 2.0, 3.0]])
    gen2 = mod.sample_sequential_z(
        species,
        box,
        num_points=4,
        bottom_z_exclusion=10.0,
        initial_pos=init_low,
        initial_types=init_types,
        seed=1,
    )
    assert (gen2[:, 2] >= 10.0 - 1e-4).all()

    # initial_types is required when initial_pos is given.
    with pytest.raises(ValueError):
        mod.sample_sequential_z(species, box, num_points=4, initial_pos=init_high)


def test_sequential_z_many_advances_independent_frontiers():
    """Multiple structures are generated independently, each with its own z."""
    mod = _small_module()
    species = torch.randint(0, 3, (7,))
    box = torch.tensor([20.0, 20.0, 120.0])
    bottom = 8.0
    many = mod.sample_sequential_z_many(
        species, box, n=4, num_points=7, bottom_z_exclusion=bottom, seed=3
    )
    assert many.shape == (4, 7, 3)
    for gen in many:
        z = gen[:, 2]
        assert (z >= bottom - 1e-4).all()
        assert (z[1:] >= z[:-1] - 1e-4).all()
    # distinct seeds -> distinct structures
    assert not torch.allclose(many[0], many[1], atol=1e-6)


def test_sequential_z_edge_cases():
    mod = _small_module()
    species = torch.randint(0, 3, (6,))
    box = torch.tensor([20.0, 20.0, 80.0])

    # Requested number of points is zero.
    out = mod.sample_sequential_z(species, box, num_points=0, seed=0)
    assert out.shape == (0, 3)

    # Only one additional molecule requested.
    one = mod.sample_sequential_z(
        species, box, num_points=1, bottom_z_exclusion=10.0, seed=0
    )
    assert one.shape == (1, 3)
    assert one[0, 2] >= 10.0 - 1e-4

    # More points than provided species -> explicit error (not silent truncation).
    with pytest.raises(ValueError):
        mod.sample_sequential_z(species, box, num_points=10)
    with pytest.raises(ValueError):
        mod.sample_sequential_z(species, box, num_points=-1)


def test_sequential_z_seeded_reproducible():
    mod = _small_module()
    species = torch.randint(0, 3, (6,))
    box = torch.tensor([20.0, 20.0, 90.0])
    g1 = mod.sample_sequential_z(
        species, box, num_points=6, bottom_z_exclusion=5.0, seed=11
    )
    g2 = mod.sample_sequential_z(
        species, box, num_points=6, bottom_z_exclusion=5.0, seed=11
    )
    assert torch.allclose(g1, g2, atol=1e-6)
    assert mod.training  # module returned to train mode after sampling


def test_sequential_z_receives_growing_structure_as_context():
    """The graph fed to the model grows by one node per generation step."""
    mod = _small_module()
    seen: list[int] = []
    orig = mod.model.forward

    def spy(*args, **kwargs):
        seen.append(int(kwargs["pos_noisy"].shape[0]))
        return orig(*args, **kwargs)

    mod.model.forward = spy
    try:
        species = torch.randint(0, 3, (5,))
        box = torch.tensor([20.0, 20.0, 100.0])
        mod.sample_sequential_z(species, box, num_points=5, steps=4, seed=0)
    finally:
        mod.model.forward = orig

    assert len(seen) == 5 * 4
    # per generation step the graph holds 1, 2, 3, 4, 5 nodes
    for k in range(5):
        assert seen[4 * k : 4 * (k + 1)] == [k + 1] * 4


def test_sequential_z_chunk_size_generates_in_blocks():
    """chunk_size > 1 reverse-diffuses several new points per step."""
    mod = _small_module(z_ordered=True, chunk_size=3)
    species = torch.randint(0, 3, (6,))
    box = torch.tensor([20.0, 20.0, 100.0])
    seen: list[int] = []
    orig = mod.model.forward

    def spy(*args, **kwargs):
        seen.append(int(kwargs["pos_noisy"].shape[0]))
        return orig(*args, **kwargs)

    mod.model.forward = spy
    try:
        gen = mod.sample_sequential_z(species, box, num_points=6, steps=4, seed=0)
    finally:
        mod.model.forward = orig

    assert gen.shape == (6, 3)
    # Chunk boundaries are z-ordered: every point of chunk 2 is at/above the
    # max z of chunk 1 (within a chunk the model freely arranges the block).
    assert (gen[3:, 2] >= gen[:3, 2].max() - 1e-5).all()
    # two chunks of 3: the graph holds 3 nodes, then 6 nodes, at every step.
    assert seen == [3] * 4 + [6] * 4

    # a chunk larger than num_points is clamped to a single chunk
    mod2 = _small_module(z_ordered=True, chunk_size=10)
    gen2 = mod2.sample_sequential_z(species, box, num_points=6, steps=2, seed=0)
    assert gen2.shape == (6, 3)


# --------------------------------------------------------------------------- #
# Model per-node time conditioning
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# z-ordered training corruption (matches the sequential sampler)
# --------------------------------------------------------------------------- #
def test_z_ordered_corrupt_keeps_context_clean():
    mod = _small_module(z_ordered=True)
    b, _box = _toy_batch()
    x_noisy, edge_index, t, eps, mask = mod._corrupt_z_ordered(b)
    clean = ~mask
    # context molecules below the studied molecule stay clean
    assert torch.allclose(x_noisy[clean], b.pos[clean], atol=1e-6)
    # targets are noised, context carries no noise
    assert eps[clean].abs().sum() == 0
    assert eps[mask].abs().sum() > 0
    assert (x_noisy[mask] != b.pos[mask]).any()
    # every graph has at least one context and one target molecule
    for g in range(b.num_graphs):
        idx = b.batch == g
        assert mask[idx].any() and (~mask[idx]).any()
    # the target (studied) molecule is the highest-z one of each graph
    for g in range(b.num_graphs):
        idx = b.batch == g
        assert b.pos[~mask & idx, 2].max() <= b.pos[mask & idx, 2].min() + 1e-6
    assert t.shape == (b.num_graphs,)
    assert x_noisy.shape == b.pos.shape
    assert edge_index.shape[0] == 2
    assert torch.isfinite(x_noisy).all()


def test_z_ordered_corrupt_uses_dataset_target_block():
    """The trainer noises exactly the dataset-provided target_mask (a block)."""
    b, _box = _toy_batch()  # each graph: z = [5, 8, 20, 35, 37, 39]
    # Give each graph a 3-node target block, as ZOrderedBoxMolecularDataset
    # with chunk_size=3 would: the three highest-z molecules.
    block = torch.zeros(len(b.pos), dtype=torch.bool)
    for g in range(b.num_graphs):
        idx = b.batch == g
        nodes = idx.nonzero().flatten()
        z = b.pos[nodes, 2]
        order = torch.argsort(z, descending=True)
        block[nodes[order[:3]]] = True
    b.target_mask = block

    mod = _small_module(z_ordered=True)
    x_noisy, edge_index, t, eps, mask = mod._corrupt_z_ordered(b)
    for g in range(b.num_graphs):
        idx = b.batch == g
        assert int(mask[idx].sum()) == 3
        # the target block is the highest-z one; context below stays clean.
        assert b.pos[~mask & idx, 2].max() <= b.pos[mask & idx, 2].min() + 1e-6
        assert torch.allclose(x_noisy[~mask & idx], b.pos[~mask & idx], atol=1e-6)
    assert eps[~mask].abs().sum() == 0
    assert eps[mask].abs().sum() > 0
    assert t.shape == (b.num_graphs,)


def test_z_ordered_training_loss_over_targets(monkeypatch):
    mod = _small_module(z_ordered=True)
    b, _box = _toy_batch()
    x_noisy, edge_index, t, eps, mask = mod._corrupt_z_ordered(b)

    # Fix the (random) corruption so training_step uses exactly these values.
    monkeypatch.setattr(
        mod, "_corrupt_z_ordered", lambda batch: (x_noisy, edge_index, t, eps, mask)
    )
    loss = mod.training_step(b, 0)
    eps_hat = mod._predict_eps(b, x_noisy, edge_index, t)
    expected = torch.nn.functional.mse_loss(eps_hat[mask], eps[mask])
    full = torch.nn.functional.mse_loss(eps_hat, eps)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert torch.allclose(loss.detach(), expected.detach(), atol=1e-6)
    # the loss uses the target mask — not the whole-graph MSE
    assert not torch.allclose(loss.detach(), full.detach(), atol=1e-6)


def test_z_ordered_eval_step_runs_and_full_mode_unchanged():
    mod = _small_module(z_ordered=True)
    b, _box = _toy_batch()
    mod._eval_step(b, "val")  # must not raise

    # default (z_ordered=False) corruption is the original 4-tuple behaviour
    mod2 = _small_module(z_ordered=False)
    x_noisy, edge_index, t, eps = mod2._corrupt(b)
    assert x_noisy.shape == b.pos.shape and t.shape == (b.num_graphs,)
    loss = mod2.training_step(b, 0)
    assert torch.isfinite(loss)
