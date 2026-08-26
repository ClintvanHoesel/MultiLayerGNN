"""Tests for the surrounding-molecule "context" mode of the scalar GNN.

A per-molecule sample's graph can additionally contain the atoms of the query
molecule's surrounding molecules (radius neighbours / k-NN / the whole box).
The model runs a per-molecule readout and returns only the query molecule's
prediction, so surrounding molecules are passed through message passing but
never trained on. Covers:

* Context datasets: node counts (radius / knn / all modes), node-level
  ``mol_number`` / ``mol_is_query`` correctness, inter-molecular (cross-molecule)
  edges, and PyG batching of context samples.
* ``ScalarMoleculeModel`` per-molecule readout: returns ``(B, T)`` (one query
  row per sample), gradients flow, and surrounding-molecule context actually
  changes the query prediction.
* PBC minimum-image edge displacements (``min_image_disp_batched`` / PBC
  ``EdgeVectorLayer`` path).
* An end-to-end Lightning ``training_step`` / optimizer step on a context batch.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from morphology_gnn.data import (
    CombinedBoxMolecularDataset,
    BoxMolecularDataset,
)
from morphology_gnn.model.embedding import EdgeVectorLayer
from morphology_gnn.model.lightning_trainer import SimpleLightningMoleculeModule
from morphology_gnn.model.scaler_model import ScalarMoleculeModel
from morphology_gnn.radius_graph import min_image_disp_batched

N_ATOMS = 4  # atoms per molecule in the synthetic box
NCOL = 3  # columns in the molecule grid


def _make_box_file(path: str, n: int = 6, seed: int = 0) -> str:
    """Write a small box HDF5 file (one box of ``n`` C4 molecules).

    Molecules sit on a 2 x 3 grid (spacing 3.3 A) so neighbours are well
    defined for both the radius and k-NN context selectors, and neighbouring
    molecules are close enough to produce inter-molecular edges at radius 3.0.
    A scalar ``energies/HOMO`` target is stored per molecule.
    """
    rng = np.random.default_rng(seed)
    atom_dt = np.dtype([("symbol", "S2"), ("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    # Mixed elements (C, N, O, C) per molecule so atom embeddings are distinct —
    # with a single element every atom maps to the same embedding and GAT's
    # normalized attention makes the aggregated message independent of the
    # neighbour set, so context could never change the query prediction.
    symbols = [b"C", b"N", b"O", b"C"]
    atoms = np.zeros((n, N_ATOMS), dtype=atom_dt)
    coms = np.zeros((n, 3))
    for i in range(n):
        r, c = divmod(i, NCOL)
        coms[i] = [1.0 + r * 3.3, 1.0 + c * 3.3, 5.0]
        atoms["symbol"][i] = symbols
        off = rng.random((N_ATOMS, 3)) * 2.0 - 1.0
        pos = np.mod(coms[i] + off, 10.0)
        atoms["x"][i], atoms["y"][i], atoms["z"][i] = pos[:, 0], pos[:, 1], pos[:, 2]
    with h5py.File(path, "w") as hf:
        g = hf.create_group("molecules")
        g.create_dataset("atoms", data=atoms)
        g.create_dataset("position", data=coms)
        g.create_dataset("orientation", data=rng.random((n, 3)))
        g.create_dataset("species", data=np.zeros(n, dtype=np.int64))
        g.create_dataset("lattice", data=np.diag([10.0, 10.0, 10.0]))
        hf.create_dataset("energies/HOMO", data=rng.normal(size=n))
        hf.create_dataset("species/name", data=np.array([b"C4"]))
        hf.create_dataset("species/smiles", data=np.array([b"CCCC"]))
    return path


# --------------------------------------------------------------------------- #
# Dataset: context graphs
# --------------------------------------------------------------------------- #
def test_context_all_node_count_and_masking(tmp_path):
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=6)
    ds = BoxMolecularDataset(
        p, target_key="HOMO", radius=3.0, keep_in_memory=True, context={"mode": "all"}
    )
    assert len(ds) == 6
    d = ds[0]
    box_atoms = sum(len(np.asarray(ds._atoms_list()[j])) for j in range(len(ds)))
    assert d.num_nodes == box_atoms == 6 * N_ATOMS
    assert d.n_atoms == N_ATOMS  # query molecule
    assert d.n_context_molecules == 5  # every other molecule in the box
    assert d.n_context_atoms == 5 * N_ATOMS
    # mol_number: query = 0, context = 1..5 ; mol_is_query marks only query atoms
    assert d.mol_number.unique().tolist() == list(range(6))
    assert d.mol_is_query.sum() == N_ATOMS
    assert d.mol_is_query[:N_ATOMS].all()
    assert not d.mol_is_query[N_ATOMS:].any()
    # y is the query molecule's target
    with h5py.File(p, "r") as hf:
        expected = float(np.asarray(hf["energies/HOMO"][0]).reshape(-1)[0])
    assert d.y[0].item() == pytest.approx(expected)


def test_context_radius_and_knn_neighbours(tmp_path):
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=6)
    dsr = BoxMolecularDataset(
        p,
        target_key=None,
        radius=3.0,
        keep_in_memory=True,
        context={"mode": "radius", "radius": 4.0},
    )
    dr = dsr[0]
    # query never its own context; number of context molecules matches precompute
    assert all(j != 0 for j in dsr._context_neighbors[0])
    assert dr.n_context_molecules == len(dsr._context_neighbors[0])
    assert dr.mol_is_query.sum() == N_ATOMS
    assert dr.num_nodes == N_ATOMS + dr.n_context_atoms

    dsk = BoxMolecularDataset(
        p,
        target_key=None,
        radius=3.0,
        keep_in_memory=True,
        context={"mode": "knn", "k": 2},
    )
    dk = dsk[0]
    assert dk.n_context_molecules == 2
    assert dk.num_nodes == N_ATOMS + 2 * N_ATOMS


def test_context_cross_molecule_edges(tmp_path):
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=6)
    ds = BoxMolecularDataset(
        p, target_key=None, radius=3.0, keep_in_memory=True, context={"mode": "all"}
    )
    d = ds[0]
    src, dst = d.edge_index
    cross = d.mol_number[src] != d.mol_number[dst]
    intra = d.mol_number[src] == d.mol_number[dst]
    assert intra.any(), "expected intra-molecular edges"
    assert cross.any(), "expected inter-molecular (context) edges"
    assert not (src == dst).any(), "no self loops"


def test_context_batches_and_collates(tmp_path):
    p1 = _make_box_file(str(tmp_path / "a.hdf5"), n=6, seed=1)
    p2 = _make_box_file(str(tmp_path / "b.hdf5"), n=5, seed=2)
    ds = CombinedBoxMolecularDataset(
        [p1, p2],
        target_key="HOMO",
        radius=3.0,
        keep_in_memory=True,
        context={"mode": "knn", "k": 3},
    )
    assert len(ds) == 11
    loader = DataLoader(ds, batch_size=4)
    batch = next(iter(loader))
    assert batch.num_graphs == 4
    assert batch.mol_number.dtype == torch.int64
    assert batch.mol_is_query.dtype == torch.bool
    assert batch.box.shape == (4, 3)
    assert batch.y.shape[0] == 4
    # exactly one query molecule per sample -> N_ATOMS query atoms each
    assert batch.mol_is_query.sum() == 4 * N_ATOMS


# --------------------------------------------------------------------------- #
# Model: per-molecule readout + PBC edge features
# --------------------------------------------------------------------------- #
def _make_context_model(**kw) -> ScalarMoleculeModel:
    defaults = dict(
        hidden_dim=16,
        num_layers=2,
        use_edge_features=True,
        pbc_edge_features=True,
        num_rbf=8,
        dropout=0.0,
        norm="GraphNorm",
    )
    defaults.update(kw)
    return ScalarMoleculeModel(**defaults)


def test_model_per_molecule_readout_shape_and_backward(tmp_path):
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=6)
    ds = BoxMolecularDataset(
        p, target_key="HOMO", radius=3.0, keep_in_memory=True, context={"mode": "all"}
    )
    loader = DataLoader(ds, batch_size=4)
    batch = next(iter(loader))
    model = _make_context_model()
    out = model(
        batch.x,
        batch.edge_index,
        batch.batch,
        batch.pos,
        mol_number=batch.mol_number,
        mol_is_query=batch.mol_is_query,
        box=batch.box,
    )
    assert out.shape == (4, 1)  # one query row per sample, despite many molecules
    out.pow(2).mean().backward()
    assert model.lin.weight.grad is not None
    assert model.atom_emb.embedding.weight.grad is not None


def test_model_requires_query_mask_with_mol_number(tmp_path):
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=6)
    ds = BoxMolecularDataset(
        p, target_key=None, radius=3.0, keep_in_memory=True, context={"mode": "all"}
    )
    d = ds[0]
    batch = Batch.from_data_list([d])
    model = _make_context_model()
    with pytest.raises(ValueError, match="mol_is_query"):
        model(
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.pos,
            mol_number=batch.mol_number,
            mol_is_query=None,
            box=batch.box,
        )


def test_context_affects_query_prediction(tmp_path):
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=6)
    ds_ctx = BoxMolecularDataset(
        p, target_key=None, radius=3.0, keep_in_memory=True, context={"mode": "all"}
    )
    ds_base = BoxMolecularDataset(p, target_key=None, radius=3.0, keep_in_memory=True)
    torch.manual_seed(0)
    model = _make_context_model()
    with torch.no_grad():
        b_ctx = Batch.from_data_list([ds_ctx[0]])
        out_ctx = model(
            b_ctx.x,
            b_ctx.edge_index,
            b_ctx.batch,
            b_ctx.pos,
            mol_number=b_ctx.mol_number,
            mol_is_query=b_ctx.mol_is_query,
            box=b_ctx.box,
        )
        b_base = Batch.from_data_list([ds_base[0]])
        out_base = model(
            b_base.x, b_base.edge_index, b_base.batch, b_base.pos, box=b_base.box
        )
    assert out_ctx.shape == (1, 1) and out_base.shape == (1, 1)
    assert not torch.allclose(
        out_ctx, out_base, atol=1e-6
    ), "surrounding-molecule context should change the query prediction"


def test_min_image_disp_batched():
    # Two atoms 9.5 A apart in a 10 A box: minimum image is -0.5, not +9.5.
    pos = torch.tensor([[0.1, 0.0, 0.0], [9.6, 0.0, 0.0]], dtype=torch.float)
    edge_index = torch.tensor([[0], [1]])
    box = torch.tensor([10.0, 10.0, 10.0]).repeat(2, 1)
    disp = min_image_disp_batched(pos, edge_index, box)
    assert torch.allclose(disp[0], torch.tensor([-0.5, 0.0, 0.0]), atol=1e-5)


def test_edge_layer_pbc_path():
    layer = EdgeVectorLayer(num_rbf=8, rbf_kwargs={"cutoff_upper": 5.0})
    pos = torch.tensor([[0.1, 0.0, 0.0], [9.6, 0.0, 0.0]], dtype=torch.float)
    edge_index = torch.tensor([[0], [1]])
    box = torch.tensor([10.0, 10.0, 10.0]).repeat(2, 1)
    pbc_attr = layer(pos, edge_index, box_per_node=box)  # min image: 0.5 A
    raw_attr = layer(pos, edge_index)  # raw displacement: 9.5 A
    # 9.5 A is beyond the 5.0 A cutoff (edge features decay to ~0), while the
    # PBC 0.5 A distance is well inside it -> the embeddings must differ.
    assert not torch.allclose(pbc_attr, raw_attr)


# --------------------------------------------------------------------------- #
# End-to-end Lightning step
# --------------------------------------------------------------------------- #
def test_context_training_step(tmp_path):
    p1 = _make_box_file(str(tmp_path / "a.hdf5"), n=6, seed=1)
    p2 = _make_box_file(str(tmp_path / "b.hdf5"), n=6, seed=2)
    ds = CombinedBoxMolecularDataset(
        [p1, p2],
        target_key="HOMO",
        radius=3.0,
        keep_in_memory=True,
        context={"mode": "radius", "radius": 4.0},
    )
    loader = DataLoader(ds, batch_size=4)
    batch = next(iter(loader))
    model = _make_context_model()
    module = SimpleLightningMoleculeModule(
        model, lr=1e-3, target_mean=torch.zeros(1), target_std=torch.ones(1)
    )
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    opt = module.configure_optimizers()
    if isinstance(opt, dict):
        opt = opt["optimizer"]
    opt.zero_grad()
    loss.backward()
    opt.step()
