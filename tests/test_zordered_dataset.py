"""Tests for :class:`ZOrderedBoxMolecularDataset`.

Covers:
* One sample per molecule; the studied molecule is marked by a per-node
  ``target_mask`` and every molecule HIGHER in z than it is thrown away.
* Sample size grows with the studied molecule's z (the highest sample holds the
  whole box, the lowest is a single molecule).
* Batching collates ``target_mask`` correctly.
* The standard random train/val/test split draws random molecules across the
  whole film (the val/test z-ranges span the film, not a single z-slab).
* ``box_sample`` / ``box_reference`` / ``n_boxes`` expose the full box for
  generation.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch

from morphology_gnn.data import ZOrderedBoxMolecularDataset


def _make_box_file(path: str, n: int = 6, seed: int = 0) -> str:
    """Small box HDF5 with ``n`` molecules whose COM z's are distinct/ascending.

    Molecule ``i`` sits at z = i + 1 (x/y random), so the z-order is the index
    order and the z-ordered samples are deterministic and easy to reason about.
    """
    rng = np.random.default_rng(seed)
    atom_dt = np.dtype([("symbol", "S2"), ("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    atoms = np.zeros((n, 4), dtype=atom_dt)
    atoms["symbol"] = b"C"
    atoms["x"] = 1.0
    atoms["y"] = 2.0
    atoms["z"] = 3.0
    bond_dt = np.dtype([("atom_1", "<i8"), ("atom_2", "<i8"), ("bond_order", "<f8")])
    bonds = np.zeros((n, 3), dtype=bond_dt)
    bonds["atom_1"] = [1, 2, 3]
    bonds["atom_2"] = [2, 3, 4]
    bonds["bond_order"] = [1.0, 1.5, 1.0]
    coms = np.stack(
        [
            rng.random(n) * 8.0,
            rng.random(n) * 8.0,
            np.arange(1, n + 1, dtype=float),  # distinct, ascending z
        ],
        axis=1,
    )
    with h5py.File(path, "w") as hf:
        g = hf.create_group("molecules")
        g.create_dataset("atoms", data=atoms)
        g.create_dataset("bonds", data=bonds)
        g.create_dataset("position", data=coms)
        g.create_dataset("orientation", data=rng.random((n, 3)))
        g.create_dataset("species", data=np.zeros(n, dtype=np.int64))
        g.create_dataset("lattice", data=np.diag([10.0, 10.0, 10.0]))
        hf.create_dataset("species/name", data=np.array([b"C6H6"]))
        hf.create_dataset("species/smiles", data=np.array([b"c1ccccc1"]))
    return path


def test_zordered_samples_throw_away_higher_molecules(tmp_path):
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=8, seed=1)
    ds = ZOrderedBoxMolecularDataset(p, radius=10.0)
    assert len(ds) == 8
    assert ds.n_boxes() == 1
    for idx in range(len(ds)):
        d = ds[idx]
        z_target = float(d.pos[d.target_mask][0, 2])
        # nothing higher in z than the studied molecule survives
        assert (d.pos[:, 2] <= z_target + 1e-6).all()
        # exactly one studied (target) molecule per sample
        assert int(d.target_mask.sum()) == 1
        assert d.box.shape == (1, 3)
        assert d.lattice.shape == (3, 3)
        assert d.edge_index.shape[0] == 2


def test_zordered_sample_sizes_grow_with_z(tmp_path):
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=8, seed=2)
    ds = ZOrderedBoxMolecularDataset(p, radius=10.0)
    sizes = [int(ds[i].pos.shape[0]) for i in range(len(ds))]
    # z is the index order, so molecule i is the (i+1)-th lowest: it keeps i+1
    # molecules (itself + everything below).
    assert sizes == [i + 1 for i in range(len(ds))]
    assert min(sizes) == 1  # lowest molecule alone
    assert max(sizes) == len(ds)  # highest molecule keeps the whole box


def test_zordered_target_mask_collates_in_batch(tmp_path):
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=8, seed=3)
    ds = ZOrderedBoxMolecularDataset(p, radius=10.0)
    batch = Batch.from_data_list([ds[0], ds[4], ds[7]])
    assert batch.target_mask.shape[0] == batch.pos.shape[0]
    assert [
        int(batch.target_mask[batch.batch == g].sum()) for g in range(batch.num_graphs)
    ] == [1, 1, 1]


def test_zordered_val_test_random_molecules_across_film(tmp_path):
    """A random per-molecule split puts val/test molecules across the film."""
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=48, seed=4)
    ds = ZOrderedBoxMolecularDataset(p, radius=10.0)
    n = len(ds)
    n_val = n // 4
    gen = torch.Generator().manual_seed(0)
    _train, val, test = torch.utils.data.random_split(
        ds, [n - 2 * n_val, n_val, n_val], generator=gen
    )
    all_z = [float(ds[i].pos[ds[i].target_mask][0, 2]) for i in range(len(ds))]
    lo, hi = min(all_z), max(all_z)
    mid = 0.5 * (lo + hi)

    for name, subset in [("val", val), ("test", test)]:
        zs = [float(ds[i].pos[ds[i].target_mask][0, 2]) for i in subset.indices]
        # val/test are not a z-slab: they contain molecules on BOTH sides of the
        # film's mid-z (a random-across-the-film draw), and cover most of it.
        below = sum(z < mid for z in zs)
        above = sum(z > mid for z in zs)
        assert (
            below >= 1 and above >= 1
        ), f"{name} sits on one side of the film (below={below}, above={above})"
        assert (max(zs) - min(zs)) > 0.4 * (
            hi - lo
        ), f"{name} z-range too narrow: {min(zs):.2f}..{max(zs):.2f}"


def test_zordered_box_generation_helpers(tmp_path):
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=6, seed=5)
    ds = ZOrderedBoxMolecularDataset(p, radius=10.0)
    full = ds.box_sample(0, radius=10.0)
    assert full.pos.shape == (6, 3)  # the full box (all molecules)
    assert not hasattr(full, "target_mask")  # no single target in the full box
    ref = ds.box_reference(0)
    assert ref["com"].shape == (6, 3)
