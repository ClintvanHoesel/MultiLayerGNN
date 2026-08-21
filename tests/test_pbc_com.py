"""Tests for the PBC-aware center-of-mass / unwrap helpers.

Covers:
* ``unwrap_molecule`` pulls atoms that straddle the periodic boundary back into
  a contiguous arrangement (reference-image path) with hand-computed positions.
* The bonds-BFS unwrap recovers a connected chain across the boundary.
* ``pbc_center_of_mass`` mass-weighted result matches a hand-computed COM, and
  is pulled toward the heavier atom.
* The wrapped COM is invariant to the unwrap strategy (reference-image vs bonds).
* ``molecule_center_of_mass`` reproduces ``molecules/position`` in the real
  SCM-pure HDF5 files (skip-if-no-data, same pattern as test_diffusion_model.py).
"""

from __future__ import annotations

import glob
import os

import h5py
import numpy as np
import pytest
import torch

from morphology_gnn.data import _atomic_mass, molecule_center_of_mass
from morphology_gnn.radius_graph import pbc_center_of_mass, unwrap_molecule

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCM_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "data_SCM_pure", "*.hdf5")))
requires_data = pytest.mark.skipif(
    not SCM_FILES, reason="no SCM-pure hdf5 files found in data/data_SCM_pure/"
)


# --------------------------------------------------------------------------- #
# Reference-image unwrap
# --------------------------------------------------------------------------- #
def test_unwrap_reference_image_pulls_straddling_atoms_together():
    """Atoms across the boundary are pulled next to the wrapped centroid."""
    box = torch.tensor([10.0, 10.0, 10.0])
    # A chain along x straddling the box: wrapped A=9.5, B=0.7, C=1.9.
    pos = torch.tensor([[9.5, 5.0, 5.0], [0.7, 5.0, 5.0], [1.9, 5.0, 5.0]])
    unwrapped = unwrap_molecule(pos, box)
    # centroid x = (9.5 + 0.7 + 1.9)/3 = 4.0333...; each atom lands at the
    # minimum-image position relative to it: x = [-0.5, 0.7, 1.9].
    assert torch.allclose(
        unwrapped[:, 0], torch.tensor([-0.5, 0.7, 1.9]), atol=1e-5
    )
    assert torch.allclose(unwrapped[:, 1:], torch.full((3, 2), 5.0), atol=1e-6)


def test_unwrap_bonds_reconnects_chain_across_boundary():
    """Walking the bond graph places the chain contiguously on one side."""
    box = torch.tensor([10.0, 10.0, 10.0])
    pos = torch.tensor([[9.5, 5.0, 5.0], [0.7, 5.0, 5.0], [1.9, 5.0, 5.0]])
    bonds = torch.tensor([[0, 1], [1, 2]])
    unwrapped = unwrap_molecule(pos, box, bonds=bonds)
    # BFS from atom 0: A stays at 9.5, B -> 10.7, C -> 11.9 (spacing 1.2).
    assert torch.allclose(
        unwrapped[:, 0], torch.tensor([9.5, 10.7, 11.9]), atol=1e-5
    )


def test_wrapped_com_invariant_to_unwrap_strategy():
    """Both unwraps fold to the same in-cell COM for a single spanning chain."""
    box = torch.tensor([10.0, 10.0, 10.0])
    pos = torch.tensor([[9.5, 5.0, 5.0], [0.7, 5.0, 5.0], [1.9, 5.0, 5.0]])
    bonds = torch.tensor([[0, 1], [1, 2]])
    com_ref = pbc_center_of_mass(pos, box, masses=None, bonds=None)
    com_bond = pbc_center_of_mass(pos, box, masses=None, bonds=bonds)
    assert torch.allclose(com_ref, com_bond, atol=1e-5)
    # mean of unwrapped x is 0.7 in-cell (see above).
    assert torch.allclose(com_ref, torch.tensor([0.7, 5.0, 5.0]), atol=1e-5)


# --------------------------------------------------------------------------- #
# Mass weighting
# --------------------------------------------------------------------------- #
def test_pbc_center_of_mass_mass_weighted():
    """A heavier atom pulls the COM toward it (fully-inside molecule)."""
    box = torch.tensor([10.0, 10.0, 10.0])
    pos = torch.tensor([[2.0, 5.0, 5.0], [3.0, 5.0, 5.0]])
    m_h = _atomic_mass("H")
    m_c = _atomic_mass("C")
    com = pbc_center_of_mass(pos, box, masses=torch.tensor([m_h, m_c]))
    expected_x = (m_h * 2.0 + m_c * 3.0) / (m_h + m_c)
    assert torch.allclose(com[0], torch.tensor(expected_x), atol=1e-5)
    # The centroid (equal weights) would be 2.5; COM sits closer to C at x=3.
    assert 2.5 < com[0].item() < 3.0


def test_pbc_center_of_mass_none_masses_is_centroid():
    box = torch.tensor([10.0, 10.0, 10.0])
    pos = torch.tensor([[2.0, 4.0, 6.0], [4.0, 4.0, 6.0]])
    com = pbc_center_of_mass(pos, box, masses=None)
    assert torch.allclose(com, torch.tensor([3.0, 4.0, 6.0]), atol=1e-6)


def test_pbc_center_of_mass_accepts_general_lattice():
    """A non-orthorhombic lattice still wraps the COM into the cell."""
    lattice = torch.tensor([[10.0, 2.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 14.0]])
    pos = torch.tensor([[9.0, 5.0, 5.0], [1.0, 7.0, 6.0]])  # fully inside
    com = pbc_center_of_mass(pos, lattice, masses=None)
    assert torch.allclose(com, torch.tensor([5.0, 6.0, 5.5]), atol=1e-5)


# --------------------------------------------------------------------------- #
# Real SCM data
# --------------------------------------------------------------------------- #
@requires_data
def test_molecule_center_of_mass_matches_stored_positions():
    """``molecule_center_of_mass`` reproduces ``molecules/position`` (both unwraps)."""
    path = SCM_FILES[0]
    with h5py.File(path, "r") as hf:
        lattice = torch.tensor(hf["molecules/lattice"][:], dtype=torch.float)
        stored = torch.tensor(hf["molecules/position"][:], dtype=torch.float)
        n = int(len(hf["molecules/atoms"]))
        for i in range(n):
            atoms = np.asarray(hf["molecules/atoms"][i])
            bonds = np.asarray(hf["molecules/bonds"][i])
            com = molecule_center_of_mass(atoms, lattice)
            com_bond = molecule_center_of_mass(atoms, lattice, bonds=bonds)
            assert torch.allclose(com, stored[i], atol=1e-3), f"mol {i} ref-image"
            assert torch.allclose(com_bond, stored[i], atol=1e-3), f"mol {i} bonds"


@requires_data
def test_scm_files_are_orthorhombic():
    """The SCM-pure cells are orthorhombic (diffusion box-length assumption)."""
    for path in SCM_FILES:
        with h5py.File(path, "r") as hf:
            lat = np.asarray(hf["molecules/lattice"][:])
        assert np.allclose(lat, np.diag(np.diagonal(lat))), path
