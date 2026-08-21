"""Tests for the SCM-pure "box" diffusion dataset.

Covers:
* ``SCMDiffusionDataset`` sample shapes and fields (COM positions, species,
  PBC radius graph, orthorhombic box).
* ``CombinedSCMDiffusionDataset`` across several files + per-box ``mol_ids``.
* ``box`` collation to ``(B, 3)`` under PyG batching.
* ``box_reference`` per-molecule metadata accessor.
* COM fallback when ``molecules/position`` is absent (computed from atoms).
* A small ``DiffusionMoleculeModule`` ``_corrupt`` / forward / ``sample_many``
  pass over batched SCM box data (finite, in-cell).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from morphology_gnn.data import (
    CombinedSCMDiffusionDataset,
    SCMDiffusionDataset,
    molecule_center_of_mass,
)
from morphology_gnn.model.diffusion_model import DiffusionMoleculeModel
from morphology_gnn.model.diffusion_trainer import DiffusionMoleculeModule


def _make_scm_file(path, n: int = 5, seed: int = 0, include_position: bool = True) -> str:
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
    bond_dt = np.dtype(
        [("atom_1", "<i8"), ("atom_2", "<i8"), ("bond_order", "<f8")]
    )
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
def test_scm_diffusion_dataset_shapes(scm_file):
    ds = SCMDiffusionDataset(scm_file, target_key=None, radius=10.0)
    assert len(ds) == 1
    assert ds.mol_ids() == ["box"]
    d = ds[0]
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


def test_combined_dataset_and_mol_ids(tmp_path):
    p1 = _make_scm_file(str(tmp_path / "a.hdf5"), n=6, seed=1)
    p2 = _make_scm_file(str(tmp_path / "b.hdf5"), n=5, seed=2)
    ds = CombinedSCMDiffusionDataset([p1, p2], target_key=None, radius=10.0)
    assert len(ds) == 2
    assert ds.mol_ids() == ["0:a", "1:b"]
    assert ds[0].pos.shape == (6, 3)
    assert ds[1].pos.shape == (5, 3)


def test_box_collates_to_batch(tmp_path):
    p1 = _make_scm_file(str(tmp_path / "a.hdf5"), n=6, seed=1)
    p2 = _make_scm_file(str(tmp_path / "b.hdf5"), n=5, seed=2)
    ds = CombinedSCMDiffusionDataset([p1, p2], target_key=None, radius=10.0)
    batch = Batch.from_data_list([ds[0], ds[1]])
    assert batch.box.shape == (2, 3)
    assert batch.num_graphs == 2
    assert batch.pos.shape == (11, 3)


def test_box_reference_accessor(scm_file):
    ds = SCMDiffusionDataset(scm_file, target_key=None, radius=10.0)
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
    ds = SCMDiffusionDataset(p, target_key=None, radius=10.0)
    d = ds[0]
    # every molecule's 4 atoms sit at (1, 2, 3): mass-weighted COM == (1, 2, 3).
    assert torch.allclose(d.pos[0], torch.tensor([1.0, 2.0, 3.0]), atol=1e-4)


def test_synthetic_com_matches_stored(tmp_path):
    p = _make_scm_file(str(tmp_path / "a.hdf5"), n=4, seed=3)
    with h5py.File(p, "r") as hf:
        lattice = torch.tensor(hf["molecules/lattice"][:], dtype=torch.float)
        atoms0 = np.asarray(hf["molecules/atoms"][0])
    com = molecule_center_of_mass(atoms0, lattice)
    assert torch.allclose(com, torch.tensor([1.0, 2.0, 3.0]), atol=1e-5)


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
    ds = CombinedSCMDiffusionDataset([p1, p2], target_key=None, radius=10.0)
    batch = Batch.from_data_list([ds[0], ds[1]])

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
    ds = SCMDiffusionDataset(p, target_key=None, radius=10.0)
    d = ds[0]
    mod = _small_diffusion_module(radius=10.0)
    gen = mod.sample_many(d.x.squeeze(-1), d.box.squeeze(0), n=2, steps=5, seed=0)
    assert gen.shape == (2, 6, 3)
    assert torch.isfinite(gen).all()
    box = d.box.squeeze(0)
    assert (gen >= -1e-3).all() and (gen <= box + 1e-3).all()
