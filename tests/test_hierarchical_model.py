"""Tests for the hierarchical molecular GNN (atoms + centre-of-mass level).

Covers:

* ``AtomToCOM`` aggregation: atoms grouped by molecule, permutation invariant,
  and never mixing atoms of different molecules.
* COM graph (``build_com_graph``): correct COM node count, cutoff respected,
  and batches containing multiple molecular systems remain separated.
* ``COMToAtom``: every atom receives information from its own molecule's COM
  node and only that one.
* SE(3) invariance: rotations + translations leave the hierarchical output
  unchanged (the existing scalar GNN is invariant; the hierarchy preserves it).
* Gradient flow through atoms -> COM -> COM GNN -> atoms: every parameter
  receives a gradient.
* Backward compatibility: ``num_hierarchical_layers=0`` reproduces
  ``ScalarMoleculeModel`` exactly; ``mol_number=None`` degrades gracefully.
* Mass-weighted, PBC-aware COM positions (hand-computed reference).
* ``build_model`` config dispatch (``model.arch: hierarchical``).
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph

from morphology_gnn.data import BoxMolecularDataset
from morphology_gnn.model.hierarchical_model import (
    AtomToCOM,
    COMToAtom,
    HierarchicalMoleculeModel,
    build_com_graph,
    filter_intra_molecular_edges,
)
from morphology_gnn.model.scaler_model import ScalarMoleculeModel
from morphology_gnn.periodic_table import PT

N_ATOMS = 4  # atoms per molecule in the synthetic box
NCOL = 3  # columns in the molecule grid


def _make_box_file(path: str, n: int = 6, seed: int = 0) -> str:
    """Write a small box HDF5 file (one box of ``n`` C/N/O/C molecules).

    Molecules sit on a 2 x 3 grid (spacing 3.3 A) so neighbours are well
    defined, and a scalar ``energies/HOMO`` target is stored per molecule.
    """
    rng = np.random.default_rng(seed)
    atom_dt = np.dtype([("symbol", "S2"), ("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
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


def _random_rotation(seed: int = 0) -> torch.Tensor:
    """A random proper 3x3 rotation matrix (from a normalized quaternion)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(4, generator=g)
    q = q / q.norm()
    w, x, y, z = q.tolist()
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float,
    )


def _make_hierarchical_model(**kw) -> HierarchicalMoleculeModel:
    """A small hierarchical model for tests (invariant-safe defaults)."""
    defaults = dict(
        hidden_dim=16,
        num_layers=2,
        num_hierarchical_layers=2,
        com_cutoff=5.0,
        com_aggregation="mean",
        com_hidden_channels=16,
        com_num_layers=1,
        use_edge_features=True,
        num_rbf=8,
        dropout=0.0,
        num_targets=1,
    )
    defaults.update(kw)
    return HierarchicalMoleculeModel(**defaults)


def _make_multi_molecule_tensors(seed: int = 0):
    """Synthetic non-PBC batch of 2 graphs x 2 molecules x 4 atoms.

    Molecule origins sit at the corners of a 4-A square (well inside the
    ``com_cutoff=5.0`` default), so each graph's two molecules always have a
    COM edge; the atomistic radius used by the caller (2.0 A) keeps the
    atomistic graph purely intra-molecular. Returns raw tensors for the model
    ``forward`` call.
    """
    g = torch.Generator().manual_seed(seed)
    origins = {
        0: (0.0, 0.0, 0.0),  # graph 0, molecule 0
        1: (4.0, 0.0, 0.0),  # graph 0, molecule 1
        2: (0.0, 4.0, 0.0),  # graph 1, molecule 0
        3: (4.0, 4.0, 0.0),  # graph 1, molecule 1
    }
    pos_list, x_list, batch_list, mol_list, query_list = [], [], [], [], []
    for mol_id, o in origins.items():
        gr = 0 if mol_id < 2 else 1
        p = (
            torch.tensor(o, dtype=torch.float)
            + (torch.rand(4, 3, generator=g) - 0.5) * 0.6
        )
        pos_list.append(p)
        x_list.append(torch.randint(1, 7, (4, 1), generator=g))
        batch_list.append(torch.full((4,), gr, dtype=torch.long))
        mol_list.append(torch.full((4,), mol_id, dtype=torch.long))
        query_list.append(torch.tensor([True] * 4 if mol_id % 2 == 0 else [False] * 4))
    return (
        torch.cat(pos_list),
        torch.cat(x_list),
        torch.cat(batch_list),
        torch.cat(mol_list),
        torch.cat(query_list),
    )


# --------------------------------------------------------------------------- #
# AtomToCOM aggregation
# --------------------------------------------------------------------------- #
def test_atom_to_com_groups_by_molecule_and_is_permutation_invariant():
    torch.manual_seed(0)
    agg = AtomToCOM(hidden_dim=8, com_hidden_channels=8, aggregation="mean")
    agg.eval()
    n_per = [3, 2, 4]
    mol_key = torch.arange(3).repeat_interleave(torch.tensor(n_per))  # N = 9
    M = 3
    h = torch.randn(mol_key.numel(), 8)
    com = agg(h, mol_key, M)
    assert com.shape == (M, 8)
    assert torch.isfinite(com).all()

    # Permutation of the atoms within each molecule -> identical COM features.
    sizes = torch.tensor(n_per)
    offsets = torch.cat([torch.tensor([0]), torch.cumsum(sizes[:-1], 0)]).tolist()
    perm = torch.cat([torch.randperm(n) + off for off, n in zip(offsets, n_per)])
    com_perm = agg(h[perm], mol_key[perm], M)
    assert torch.allclose(com, com_perm, atol=1e-5)

    # No cross-molecule contamination: perturb molecule 2 only.
    h_b = h.clone()
    h_b[mol_key == 2] += 100.0
    com_b = agg(h_b, mol_key, M)
    assert torch.allclose(com_b[0], com[0], atol=1e-5)  # molecule 0 untouched
    assert torch.allclose(com_b[1], com[1], atol=1e-5)  # molecule 1 untouched
    assert not torch.allclose(com_b[2], com[2], atol=1e-2)  # molecule 2 changed

    # sum and attention are also permutation invariant and keep molecule groups.
    for aggregation in ("sum", "attention"):
        agg2 = AtomToCOM(hidden_dim=8, com_hidden_channels=8, aggregation=aggregation)
        agg2.eval()
        c = agg2(h, mol_key, M)
        assert c.shape == (M, 8)
        cp = agg2(h[perm], mol_key[perm], M)
        assert torch.allclose(
            c, cp, atol=1e-5
        ), f"{aggregation} not permutation invariant"


# --------------------------------------------------------------------------- #
# COM graph
# --------------------------------------------------------------------------- #
def test_build_com_graph_cutoff_and_batch_separation():
    box = torch.tensor([[10.0, 10.0, 10.0], [12.0, 12.0, 12.0]])
    # Graph 0: nodes 0,1 close (3.0) and node 2 far; graph 1: nodes 3,4 close.
    com_pos = torch.tensor(
        [
            [1.0, 1.0, 1.0],  # g0
            [1.0, 1.0, 4.0],  # g0
            [5.0, 5.0, 5.0],  # g0 (far)
            [1.0, 1.0, 1.0],  # g1
            [1.0, 1.0, 4.2],  # g1
        ],
        dtype=torch.float,
    )
    com_batch = torch.tensor([0, 0, 0, 1, 1])
    cutoff = 4.0
    edge = build_com_graph(com_pos, com_batch, box, cutoff)
    src, dst = edge
    assert (src != dst).all(), "no self loops"
    # Batches remain separated: no COM edge crosses graphs.
    assert (com_batch[src] == com_batch[dst]).all()
    # Cutoff respected under the minimum-image convention.
    disp = com_pos[dst] - com_pos[src]
    box_src = box[com_batch[src]]
    d = torch.norm(disp - torch.round(disp / box_src) * box_src, dim=1)
    assert (d <= cutoff + 1e-5).all()
    pairs = set(zip(src.tolist(), dst.tolist()))
    assert (0, 1) in pairs or (1, 0) in pairs  # close pair in graph 0
    assert (0, 2) not in pairs and (2, 0) not in pairs  # far pair in graph 0
    assert (3, 4) in pairs or (4, 3) in pairs  # close pair in graph 1


def test_build_com_graph_non_pbc():
    com_pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
        ],
        dtype=torch.float,
    )
    com_batch = torch.tensor([0, 0, 0, 1, 1])
    edge = build_com_graph(com_pos, com_batch, None, 2.0)
    src, dst = edge
    assert (com_batch[src] == com_batch[dst]).all()
    pairs = set(zip(src.tolist(), dst.tolist()))
    assert (0, 1) in pairs or (1, 0) in pairs
    assert (0, 2) not in pairs and (2, 0) not in pairs
    assert (3, 4) in pairs or (4, 3) in pairs


def test_filter_intra_molecular_edges():
    # 5 atoms; molecules: mol0 = {0, 1}, mol1 = {2, 3, 4}.
    mol_number = torch.tensor([0, 0, 1, 1, 1])
    # (0,1) intra, (2,0) cross, (1,3) cross, (3,4) intra, (4,2) intra.
    edge_index = torch.tensor([[0, 2, 1, 3, 4], [1, 0, 3, 4, 2]])
    out = filter_intra_molecular_edges(edge_index, mol_number)
    assert torch.equal(out, torch.tensor([[0, 3, 4], [1, 4, 2]]))


def test_build_com_graph_all_mode():
    com_pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # g0
            [1.0, 0.0, 0.0],  # g0
            [0.0, 1.0, 0.0],  # g0 (1.41 A from node 0)
            [5.0, 5.0, 5.0],  # g1
            [5.0, 6.0, 5.0],  # g1
        ],
        dtype=torch.float,
    )
    com_batch = torch.tensor([0, 0, 0, 1, 1])
    # cutoff=1.0 is below several pair distances (0<->2 is ~1.41 A): mode="all"
    # must still connect them (all molecules in the vicinity, no distance limit).
    edge = build_com_graph(com_pos, com_batch, None, 1.0, mode="all")
    src, dst = edge
    assert (com_batch[src] == com_batch[dst]).all()  # graphs stay separated
    assert (src != dst).all()  # no self-loops
    # graph 0: 3 nodes -> 3*2 directed edges; graph 1: 2 nodes -> 2 edges.
    assert int((com_batch == 0)[src].sum()) == 6
    assert int((com_batch == 1)[src].sum()) == 2
    pairs = set(zip(src.tolist(), dst.tolist()))
    for i in range(3):
        for j in range(3):
            if i != j:
                assert (i, j) in pairs  # fully connected within graph 0
    assert (3, 4) in pairs and (4, 3) in pairs
    with pytest.raises(ValueError, match="com_graph mode"):
        build_com_graph(com_pos, com_batch, None, 1.0, mode="bogus")


def test_com_node_count_equals_number_of_molecules():
    # A full hierarchical forward produces one COM row per (graph, molecule).
    model = _make_hierarchical_model()
    model.eval()
    pos, x, batch, mol_number, mol_is_query = _make_multi_molecule_tensors()
    edge_index = radius_graph(pos, r=2.0, loop=False)
    atom_mol = torch.unique(
        mol_number + batch * (mol_number.max() + 1), return_inverse=True
    )[1]
    M = int(atom_mol.max()) + 1
    com_pos, com_batch = model._compute_com_positions(pos, x, batch, None, atom_mol, M)
    assert com_pos.shape == (4, 3)  # 4 molecules across 2 graphs
    assert sorted(com_batch.tolist()) == [0, 0, 1, 1]  # 2 molecules per graph


# --------------------------------------------------------------------------- #
# COMToAtom
# --------------------------------------------------------------------------- #
def test_com_to_atom_receives_only_own_molecule():
    torch.manual_seed(0)
    c2a = COMToAtom(com_hidden_channels=6, hidden_dim=8, gated=True)
    c2a.eval()
    # atom_mol[i] = molecule of atom i: [0, 0, 1, 1, 2]
    atom_mol = torch.tensor([0, 0, 1, 1, 2])
    h_atoms = torch.randn(5, 8)
    h_com = torch.randn(3, 6)
    out = c2a(h_atoms, h_com, atom_mol)

    # Perturb only molecule 1's COM row -> atoms 0, 1 (mol 0) and 4 (mol 2)
    # unchanged; atoms 2, 3 (mol 1) change.
    h_com_pert = h_com.clone()
    h_com_pert[1] += 10.0
    out_pert = c2a(h_atoms, h_com_pert, atom_mol)
    for i in (0, 1, 4):
        assert torch.allclose(out[i], out_pert[i], atol=1e-5), f"atom {i} changed"
    for i in (2, 3):
        assert not torch.allclose(out[i], out_pert[i], atol=1e-2), f"atom {i} unchanged"

    # Non-gated variant runs and is still molecule-local.
    c2a2 = COMToAtom(com_hidden_channels=6, hidden_dim=8, gated=False)
    c2a2.eval()
    out2 = c2a2(h_atoms, h_com, atom_mol)
    assert out2.shape == out.shape
    out2_pert = c2a2(h_atoms, h_com_pert, atom_mol)
    assert torch.allclose(out2[0], out2_pert[0], atol=1e-5)
    assert not torch.allclose(out2[2], out2_pert[2], atol=1e-2)


# --------------------------------------------------------------------------- #
# SE(3) invariance
# --------------------------------------------------------------------------- #
def test_hierarchical_model_rotation_and_translation_invariance():
    model = _make_hierarchical_model()
    model.eval()
    pos, x, batch, mol_number, mol_is_query = _make_multi_molecule_tensors()
    edge_index = radius_graph(pos, r=2.0, loop=False)

    with torch.no_grad():
        out = model(
            x,
            edge_index,
            batch,
            pos,
            mol_number=mol_number,
            mol_is_query=mol_is_query,
            box=None,
        )
    assert out.shape == (2, 1)  # one query molecule per graph

    # Rotation + translation (rigid: preserves every pairwise distance).
    R = _random_rotation(seed=3)
    t = torch.tensor([0.5, -1.0, 2.0])
    pos_rt = pos @ R.t() + t
    edge_index_rt = radius_graph(pos_rt, r=2.0, loop=False)
    with torch.no_grad():
        out_rt = model(
            x,
            edge_index_rt,
            batch,
            pos_rt,
            mol_number=mol_number,
            mol_is_query=mol_is_query,
            box=None,
        )
    assert torch.allclose(out, out_rt, atol=1e-4)

    # Pure translation.
    pos_t = pos + t
    edge_index_t = radius_graph(pos_t, r=2.0, loop=False)
    with torch.no_grad():
        out_t = model(
            x,
            edge_index_t,
            batch,
            pos_t,
            mol_number=mol_number,
            mol_is_query=mol_is_query,
            box=None,
        )
    assert torch.allclose(out, out_t, atol=1e-4)


# --------------------------------------------------------------------------- #
# Gradient flow + full PBC forward
# --------------------------------------------------------------------------- #
def test_gradient_flow_through_hierarchy(tmp_path):
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=6)
    ds = BoxMolecularDataset(
        p, target_key="HOMO", radius=3.0, keep_in_memory=True, context={"mode": "all"}
    )
    loader = DataLoader(ds, batch_size=4)
    batch = next(iter(loader))
    model = _make_hierarchical_model(pbc_edge_features=True)
    out = model(
        batch.x,
        batch.edge_index,
        batch.batch,
        batch.pos,
        mol_number=batch.mol_number,
        mol_is_query=batch.mol_is_query,
        box=batch.box,
    )
    assert out.shape == (4, 1)
    out.pow(2).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters without gradient: {missing}"


def test_hierarchical_forward_shapes_on_context_batch(tmp_path):
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=6)
    ds = BoxMolecularDataset(
        p, target_key="HOMO", radius=3.0, keep_in_memory=True, context={"mode": "all"}
    )
    d = ds[0]
    batch = Batch.from_data_list([d])
    model = _make_hierarchical_model(
        num_hierarchical_layers=3, com_aggregation="attention"
    )
    model.eval()
    with torch.no_grad():
        out = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.pos,
            mol_number=batch.mol_number,
            mol_is_query=batch.mol_is_query,
            box=batch.box,
        )
    assert out.shape == (1, 1)


def test_hierarchical_intra_atomistic_edges_and_com_all(tmp_path):
    """Context on: the atomistic graph is intra-molecular only (no atom-level
    cross-talk) while the COM graph is fully connected (com_graph="all"), and
    gradients still flow through the whole atoms -> COM -> atoms path."""
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=6)
    ds = BoxMolecularDataset(
        p, target_key="HOMO", radius=3.0, keep_in_memory=True, context={"mode": "all"}
    )
    loader = DataLoader(ds, batch_size=4)
    batch = next(iter(loader))
    model = _make_hierarchical_model(
        pbc_edge_features=True, atomistic_edges="intra", com_graph="all"
    )
    out = model(
        batch.x,
        batch.edge_index,
        batch.batch,
        batch.pos,
        mol_number=batch.mol_number,
        mol_is_query=batch.mol_is_query,
        box=batch.box,
    )
    assert out.shape == (4, 1)
    out.pow(2).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters without gradient: {missing}"


# --------------------------------------------------------------------------- #
# Backward compatibility
# --------------------------------------------------------------------------- #
def test_backward_compat_num_hierarchical_layers_zero(tmp_path):
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=6)
    ds = BoxMolecularDataset(p, target_key=None, radius=3.0, keep_in_memory=True)
    batch = Batch.from_data_list([ds[0]])
    cfg = dict(
        hidden_dim=16,
        num_layers=2,
        use_edge_features=True,
        pbc_edge_features=True,
        num_rbf=8,
        dropout=0.0,
        norm="GraphNorm",
    )
    torch.manual_seed(0)
    scalar = ScalarMoleculeModel(**cfg)
    torch.manual_seed(0)
    hier = HierarchicalMoleculeModel(num_hierarchical_layers=0, **cfg)
    scalar.eval()
    hier.eval()
    with torch.no_grad():
        y_s = scalar(batch.x, batch.edge_index, batch.batch, batch.pos, box=batch.box)
        y_h = hier(batch.x, batch.edge_index, batch.batch, batch.pos, box=batch.box)
    assert y_s.shape == y_h.shape == (1, 1)
    assert torch.allclose(y_s, y_h, atol=1e-5)


def test_mol_number_none_degrades_to_per_graph(tmp_path):
    # Per-molecule samples (no context -> no mol_number): hierarchical model runs
    # with each graph treated as one molecule (per-graph readout).
    p = _make_box_file(str(tmp_path / "box.hdf5"), n=6)
    ds = BoxMolecularDataset(p, target_key=None, radius=3.0, keep_in_memory=True)
    loader = DataLoader(ds, batch_size=4)
    batch = next(iter(loader))
    model = _make_hierarchical_model(num_hierarchical_layers=2)
    model.eval()
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.batch, batch.pos, box=batch.box)
    assert out.shape == (4, 1)


# --------------------------------------------------------------------------- #
# End-to-end Lightning step
# --------------------------------------------------------------------------- #
def test_lightning_training_step(tmp_path):
    """Full training entry path: Lightning module + hierarchical model on a
    context batch (loss finite, optimizer step runs)."""
    from morphology_gnn.model.lightning_trainer import (
        SimpleLightningMoleculeModule,
    )

    p = _make_box_file(str(tmp_path / "box.hdf5"), n=6)
    ds = BoxMolecularDataset(
        p, target_key="HOMO", radius=3.0, keep_in_memory=True, context={"mode": "all"}
    )
    loader = DataLoader(ds, batch_size=4)
    batch = next(iter(loader))
    model = _make_hierarchical_model(pbc_edge_features=True)
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


# --------------------------------------------------------------------------- #
# COM positions
# --------------------------------------------------------------------------- #
def test_com_position_mass_weighted_and_pbc():
    model = _make_hierarchical_model()
    model.eval()
    box = torch.tensor([[10.0, 10.0, 10.0]])
    # Molecule 0 straddles the boundary: C at x=9.5, H at x=0.7.
    # Molecule 1 is interior.
    pos = torch.tensor(
        [
            [9.5, 5.0, 5.0],  # mol 0, Z=6 (C)
            [0.7, 5.0, 5.0],  # mol 0, Z=1 (H)
            [2.0, 2.0, 2.0],  # mol 1, Z=6
            [2.0, 2.0, 2.8],  # mol 1, Z=1
        ],
        dtype=torch.float,
    )
    z = torch.tensor([[6], [1], [6], [1]])
    batch = torch.zeros(4, dtype=torch.long)
    atom_mol = torch.tensor([0, 0, 1, 1])
    M = 2
    com_pos, com_batch = model._compute_com_positions(pos, z, batch, box, atom_mol, M)
    assert com_batch.tolist() == [0, 0]

    # Hand-computed COM of mol 0: H is unwrapped to x=10.7 (minimum image of 0.7
    # relative to the anchor at 9.5), then mass-weighted and folded into [0, 10).
    mC = PT.get_mass(6)
    mH = PT.get_mass(1)
    expected_x = (mC * 9.5 + mH * 10.7) / (mC + mH)
    assert expected_x < 10.0
    assert torch.allclose(com_pos[0], torch.tensor([expected_x, 5.0, 5.0]), atol=1e-4)

    # Mol 1: interior, plain mass-weighted mean (no PBC effect).
    expected_z1 = (mC * 2.0 + mH * 2.8) / (mC + mH)
    assert torch.allclose(com_pos[1], torch.tensor([2.0, 2.0, expected_z1]), atol=1e-5)


# --------------------------------------------------------------------------- #
# Config wiring
# --------------------------------------------------------------------------- #
def test_build_model_arch_dispatch():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = os.path.join(root, "runs")
    if runs_dir not in sys.path:
        sys.path.insert(0, runs_dir)
    try:
        from training_helpers import build_model
    except ImportError as exc:  # pragma: no cover - env-specific
        pytest.skip(f"runs/training_helpers not importable: {exc}")

    cfg = dict(
        arch="hierarchical",
        hidden_dim=8,
        num_layers=1,
        num_hierarchical_layers=2,
        com_cutoff=7.0,
        com_aggregation="sum",
        com_hidden_channels=12,
        com_num_layers=2,
        num_rbf=8,
        use_edge_features=True,
        dropout=0.0,
    )
    model = build_model(cfg, radius=5.0, context={})
    assert isinstance(model, HierarchicalMoleculeModel)
    assert model.num_hierarchical_layers == 2
    assert model.com_cutoff == 7.0
    assert model.com_aggregation == "sum"
    assert model.com_hidden_channels == 12
    assert model.com_num_layers == 2

    # Default (no arch) still builds the scalar model.
    scalar = build_model(
        dict(hidden_dim=8, num_layers=1, use_edge_features=False), radius=5.0
    )
    assert isinstance(scalar, ScalarMoleculeModel)
