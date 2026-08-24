"""Tests for SCM-pure "box" samples built from the per-molecule SCM datasets.

The diffusion runner needs no dedicated diffusion dataset class: box-level
samples (molecules as nodes — COM positions, species, PBC radius graph over
COMs) are assembled by ``SCMMolecularDataset.box_sample`` /
``CombinedSCMMolecularDataset.box_sample`` from the per-molecule SCM datasets.
Covers:
* ``box_sample`` shapes and fields (COM positions, species, PBC radius graph,
  orthorhombic box) for a single file and across several files.
* ``box`` collation to ``(B, 3)`` under PyG batching.
* ``box_reference`` per-molecule metadata accessor.
* COM fallback when ``molecules/position`` is absent (computed from atoms).
* A small ``DiffusionMoleculeModule`` ``_corrupt`` / forward / ``sample_many``
  pass over batched SCM box samples (finite, in-cell).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from morphology_gnn.data import (
    CombinedSCMMolecularDataset,
    SCMMolecularDataset,
    molecule_center_of_mass,
)
from morphology_gnn.model.diffusion_model import DiffusionMoleculeModel
from morphology_gnn.model.diffusion_trainer import DiffusionMoleculeModule


def _make_scm_file(
    path, n: int = 5, seed: int = 0, include_position: bool = True
) -> str:
    """Write a small SCM-pure HDF5 file (one box of ``n`` identical C4 molecules).

    ``molecules/atoms`` is a compound dataset of shape ``(n, 4)`` with fields
    ``symbol/x/y/z`` and ``molecules/bonds`` of shape ``(n, 3)`` with fields
    ``atom_1/atom_2/bond_order`` (1-based) — reading ``atoms[i]`` / ``bonds[i]``
    yields the same structured arrays as the real SCM-pure files.
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
    with h5py.File(path, "w") as hf:
        g = hf.create_group("molecules")
        g.create_dataset("atoms", data=atoms)
        g.create_dataset("bonds", data=bonds)
        if include_position:
            g.create_dataset("position", data=(rng.random((n, 3)) * 10.0))
        g.create_dataset("orientation", data=(rng.random((n, 3))))
        g.create_dataset("species", data=np.zeros(n, dtype=np.int64))
        g.create_dataset("lattice", data=np.diag([10.0, 10.0, 10.0]))
        hf.create_dataset("species/name", data=np.array([b"C6H6"]))
        hf.create_dataset("species/smiles", data=np.array([b"c1ccccc1"]))
    return path


@pytest.fixture
def scm_file(tmp_path) -> str:
    return _make_scm_file(str(tmp_path / "box.hdf5"), n=5, seed=1)


# --------------------------------------------------------------------------- #
# Dataset shapes / fields
# --------------------------------------------------------------------------- #
def test_scm_box_sample_shapes(scm_file):
    ds = SCMMolecularDataset(scm_file, target_key=None, radius=10.0)
    assert len(ds) == 5  # 5 molecules in the box
    assert ds.mol_name == "box"
    d = ds.box_sample(radius=10.0)
    n = 5
    assert d.pos.shape == (n, 3)
    assert d.x.shape == (n, 1)
    assert d.box.shape == (1, 3)
    assert d.lattice.shape == (3, 3)
    assert d.edge_index.shape[0] == 2
    assert d.edge_index.shape[1] > 0
    assert not (d.edge_index[0] == d.edge_index[1]).any()  # no self loops
    assert d.is_orthorhombic.tolist() == [1]
    assert d.mol_name == "box"
    # all stored COMs are inside the cell
    assert (d.pos >= 0).all() and (d.pos <= 10.0).all()


def test_combined_box_samples_across_files(tmp_path):
    p1 = _make_scm_file(str(tmp_path / "a.hdf5"), n=6, seed=1)
    p2 = _make_scm_file(str(tmp_path / "b.hdf5"), n=5, seed=2)
    ds = CombinedSCMMolecularDataset([p1, p2], target_key=None, radius=10.0)
    assert ds.n_boxes() == 2
    # per-molecule samples still exposed for the normal trainer
    assert len(ds) == 11
    assert ds.box_sample(0).pos.shape == (6, 3)
    assert ds.box_sample(1).pos.shape == (5, 3)
    assert ds.box_sample(0).mol_name == "a"
    assert ds.box_sample(1).mol_name == "b"
    # file-qualified box ids (same convention the diffusion runner uses)
    assert [f"{di}:{d.mol_name}" for di, d in enumerate(ds.datasets)] == ["0:a", "1:b"]


def test_box_collates_to_batch(tmp_path):
    p1 = _make_scm_file(str(tmp_path / "a.hdf5"), n=6, seed=1)
    p2 = _make_scm_file(str(tmp_path / "b.hdf5"), n=5, seed=2)
    ds = CombinedSCMMolecularDataset([p1, p2], target_key=None, radius=10.0)
    batch = Batch.from_data_list([ds.box_sample(0), ds.box_sample(1)])
    assert batch.box.shape == (2, 3)
    assert batch.num_graphs == 2
    assert batch.pos.shape == (11, 3)


def test_box_reference_accessor(scm_file):
    ds = SCMMolecularDataset(scm_file, target_key=None, radius=10.0)
    ref = ds.box_reference()
    assert len(ref["atoms"]) == 5
    assert ref["com"].shape == (5, 3)
    assert ref["box"].shape == (1, 3)
    assert ref["species_names"] == ["C6H6"]
    assert ref["species_smiles"] == ["c1ccccc1"]
    assert ref["orientation"].shape == (5, 3)


def test_com_fallback_when_position_missing(tmp_path):
    """Without ``molecules/position`` the dataset recomputes the PBC COM."""
    p = _make_scm_file(str(tmp_path / "a.hdf5"), n=4, seed=4, include_position=False)
    ds = SCMMolecularDataset(p, target_key=None, radius=10.0)
    d = ds.box_sample(radius=10.0)
    # every molecule's 4 atoms sit at (1, 2, 3): mass-weighted COM == (1, 2, 3).
    assert torch.allclose(d.pos[0], torch.tensor([1.0, 2.0, 3.0]), atol=1e-4)


def test_synthetic_com_matches_stored(tmp_path):
    p = _make_scm_file(str(tmp_path / "a.hdf5"), n=4, seed=3)
    with h5py.File(p, "r") as hf:
        lattice = torch.tensor(hf["molecules/lattice"][:], dtype=torch.float)
        atoms0 = np.asarray(hf["molecules/atoms"][0])
    com = molecule_center_of_mass(atoms0, lattice)
    assert torch.allclose(com, torch.tensor([1.0, 2.0, 3.0]), atol=1e-5)


def test_keep_in_memory_matches_disk(scm_file):
    """keep_in_memory=True yields identical samples to the default disk reads."""
    disk = SCMMolecularDataset(scm_file, target_key=None, radius=10.0)
    mem = SCMMolecularDataset(
        scm_file, target_key=None, radius=10.0, keep_in_memory=True
    )
    assert mem._cache is not None
    assert disk._cache is None
    for i in range(len(disk)):
        a, b = disk[i], mem[i]
        assert torch.equal(a.x, b.x)
        assert torch.allclose(a.pos, b.pos)
        assert torch.equal(a.edge_index, b.edge_index)
        assert torch.equal(a.box, b.box)
        assert a.mol_name == b.mol_name
        assert a.n_atoms == b.n_atoms
    assert torch.equal(disk.lattice, mem.lattice)
    assert torch.equal(disk.coms(), mem.coms())
    assert torch.equal(disk.species_ids(), mem.species_ids())
    assert torch.equal(disk.box_sample(radius=10.0).pos, mem.box_sample(radius=10.0).pos)
    rd, rm = disk.box_reference(), mem.box_reference()
    assert torch.equal(rd["com"], rm["com"])
    assert torch.equal(rd["box"], rm["box"])
    assert rd["species_names"] == rm["species_names"]
    assert rd["species_smiles"] == rm["species_smiles"]


def test_keep_in_memory_with_targets(tmp_path):
    """Cached per-target reads match the disk-backed path."""
    p = _make_scm_file(str(tmp_path / "t.hdf5"), n=4, seed=0)
    with h5py.File(p, "a") as hf:
        hf.create_dataset("energies/HOMO", data=np.arange(4, dtype=np.float64))
    disk = SCMMolecularDataset(p, target_key="HOMO", radius=10.0)
    mem = SCMMolecularDataset(p, target_key="HOMO", radius=10.0, keep_in_memory=True)
    assert torch.allclose(disk._target_values(), mem._target_values())
    assert torch.allclose(disk.target_mean_std()[0], mem.target_mean_std()[0])
    assert torch.allclose(disk[0].y, mem[0].y)


# --------------------------------------------------------------------------- #
# Diffusion integration (small model on CPU)
# --------------------------------------------------------------------------- #
def _small_diffusion_module(radius: float = 10.0) -> DiffusionMoleculeModule:
    model = DiffusionMoleculeModel(
        hidden_dim=16, num_layers=2, num_rbf=8, cell_embed_dim=4, dropout=0.0
    )
    return DiffusionMoleculeModule(model, radius=radius)


def test_corrupt_and_predict_on_batched_boxes(tmp_path):
    p1 = _make_scm_file(str(tmp_path / "a.hdf5"), n=6, seed=1)
    p2 = _make_scm_file(str(tmp_path / "b.hdf5"), n=5, seed=2)
    ds = CombinedSCMMolecularDataset([p1, p2], target_key=None, radius=10.0)
    batch = Batch.from_data_list([ds.box_sample(0), ds.box_sample(1)])

    mod = _small_diffusion_module(radius=10.0)
    x_noisy, edge_index, t, eps = mod._corrupt(batch)
    assert x_noisy.shape == batch.pos.shape
    assert t.shape == (2,)
    assert eps.shape == batch.pos.shape
    assert torch.isfinite(x_noisy).all()

    eps_hat = mod._predict_eps(batch, x_noisy, edge_index, t)
    assert eps_hat.shape == batch.pos.shape
    assert torch.isfinite(eps_hat).all()
    loss = torch.nn.functional.mse_loss(eps_hat, eps)
    assert loss.shape == () and torch.isfinite(loss)


def test_sample_many_in_cell(tmp_path):
    p = _make_scm_file(str(tmp_path / "a.hdf5"), n=6, seed=1)
    ds = SCMMolecularDataset(p, target_key=None, radius=10.0)
    d = ds.box_sample(radius=10.0)
    mod = _small_diffusion_module(radius=10.0)
    gen = mod.sample_many(d.x.squeeze(-1), d.box.squeeze(0), n=2, steps=5, seed=0)
    assert gen.shape == (2, 6, 3)
    assert torch.isfinite(gen).all()
    box = d.box.squeeze(0)
    assert (gen >= -1e-3).all() and (gen <= box + 1e-3).all()
