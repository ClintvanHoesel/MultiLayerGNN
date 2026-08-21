import logging
import os
from typing import Union

import h5py
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import radius_graph

from .periodic_table import PT
from .radius_graph import (
    _normalize_lattice,
    pbc_center_of_mass,
    radius_graph_pbc,
    unwrap_molecule,
)

logger = logging.getLogger(__name__)

try:
    from .cuda_radius_graph import radius_graph_pbc as cuda_radius_graph_pbc

    _cuda_available = True
    logger.debug("CUDA radius graph extension available")
except Exception as exc:
    cuda_radius_graph_pbc = None
    _cuda_available = False
    logger.warning(
        "CUDA radius graph unavailable (%s); using the Python implementation", exc
    )


class H5MolecularDataset(Dataset):
    """
    A PyTorch Geometric Dataset that loads molecular graphs directly from HDF5 files.
    This implementation reconstructs the graph structure (edge_index) on-the-fly
    making it extremely memory efficient.

    Two HDF5 layouts are supported per top-level group:

    * ``pos`` of shape ``(N, 3)``: the group is a single molecule.
    * ``pos`` of shape ``(frames, N, 3)`` (e.g. MD trajectories): each frame is
      exposed as its own sample.
    """

    def __init__(
        self,
        h5_path: str,
        target_key: str,
        radius: float = 6.0,
        box_key: str = "lattice",
    ):
        """
        Args:
            h5_path (str): Path to the HDF5 file.
            target_key (str | list[str]): The key(s) in the HDF5 group containing the target
                property/properties (e.g. 'Positive VIP' or ['Positive VIP', 'HOMO']).
            radius (float): The radius for radius_graph construction.
            box_key (str): The key in the HDF5 group containing the periodic box lattice.
        """
        self.h5_path = h5_path
        self.target_keys = (
            [target_key] if isinstance(target_key, str) else list(target_key)
        )
        if not self.target_keys:
            raise ValueError("target_key must be a non-empty string or list of strings")
        self.radius = radius
        self.box_key = box_key

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found at: {h5_path}")

        # Build a flat index of (group key, frame index) samples and validate
        # that every requested target key exists somewhere in the file.
        self._index: list[tuple[str, int | None]] = []
        seen_keys: set[str] = set()
        with h5py.File(self.h5_path, "r") as hf:
            for key in hf.keys():
                group = hf[key]
                if not isinstance(group, h5py.Group) or "pos" not in group:
                    continue
                seen_keys.update(group.keys())
                if group["pos"].ndim == 3:
                    n_frames = group["pos"].shape[0]
                    self._index.extend((key, f) for f in range(n_frames))
                else:
                    self._index.append((key, None))

        missing = [tk for tk in self.target_keys if tk not in seen_keys]
        if missing:
            raise KeyError(
                f"target key(s) {missing} not found in {h5_path}; "
                f"available keys include {sorted(seen_keys)}"
            )

        self._num_samples = len(self._index)
        logger.info(
            "H5MolecularDataset %s: %d sample(s), radius=%.3f, targets=%r",
            h5_path,
            self._num_samples,
            radius,
            self.target_keys,
        )
        super().__init__(root=None)

    def __len__(self) -> int:
        return self._num_samples

    def mol_ids(self) -> list[str]:
        """Molecule identifier per sample (for group-aware cross-validation).

        Every top-level HDF5 group is treated as one molecule; samples from a
        group with ``(frames, N, 3)`` positions (trajectories) share the id.
        """
        return [mol_key for mol_key, _ in self._index]

    def _target_values(self, indices=None) -> torch.Tensor:
        """Stack the target vector for the given sample indices: ``(n, num_targets)``.

        Reads only the target arrays from HDF5 (no graph construction), so
        computing mean/std is cheap.
        """
        if indices is None:
            indices = range(len(self))
        rows = []
        with h5py.File(self.h5_path, "r") as hf:
            for i in indices:
                mol_key, frame = self._index[i]
                group = hf[mol_key]
                vals = []
                for tk in self.target_keys:
                    raw = group[tk][frame] if frame is not None else group[tk][:]
                    vals.append(torch.as_tensor(raw).reshape(-1)[0].item())
                rows.append(vals)
        return torch.tensor(rows, dtype=torch.float)

    def target_mean_std(self, indices=None):
        """Per-target mean/std over the (optionally subset) samples.

        Returns ``(mean, std)`` tensors of shape ``(num_targets,)``; zero-
        variance columns get ``std = 1`` so standardization stays well-defined.
        Used to standardize targets (fit on the training split only).
        """
        y = self._target_values(indices)
        mean = y.mean(dim=0)
        std = y.std(dim=0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return mean, std

    @property
    def target_stats(self):
        """Per-target ``(mean, std)`` over all samples of this dataset."""
        return self.target_mean_std()

    def __getitem__(self, idx: int) -> Data:
        """
        Loads a single molecule frame's data and constructs the graph.
        """
        mol_key, frame = self._index[idx]
        logger.debug("loading sample idx=%d mol=%s frame=%s", idx, mol_key, frame)

        with h5py.File(self.h5_path, "r") as hf:
            group = hf[mol_key]

            # Load positions and atom types for a single frame.
            if frame is not None:
                pos = torch.tensor(group["pos"][frame], dtype=torch.float)
                types = torch.tensor(group["types"][frame], dtype=torch.long)
            else:
                pos = torch.tensor(group["pos"][:], dtype=torch.float)
                types = torch.tensor(group["types"][:], dtype=torch.long)

            # Load one or several target properties (multi-target training).
            ys = []
            for tk in self.target_keys:
                if frame is not None:
                    ys.append(torch.tensor(group[tk][frame], dtype=torch.float))
                else:
                    ys.append(torch.tensor(group[tk][:], dtype=torch.float))
            y = torch.cat([yi.view(-1) for yi in ys])  # (num_targets,)

            # Load box lattice if available and build a periodic radius graph.
            lattice = None
            if self.box_key in group:
                lattice = torch.tensor(group[self.box_key][:], dtype=torch.float)

            if lattice is not None:
                if cuda_radius_graph_pbc is not None and _cuda_available:
                    try:
                        edge_index = cuda_radius_graph_pbc(
                            pos,
                            r=self.radius,
                            lattice=lattice,
                            loop=False,
                        )
                    except Exception:
                        edge_index = radius_graph_pbc(
                            pos, r=self.radius, lattice=lattice, loop=False
                        )
                else:
                    edge_index = radius_graph_pbc(
                        pos, r=self.radius, lattice=lattice, loop=False
                    )
            else:
                edge_index = radius_graph(pos, r=self.radius, loop=False)

            # Create the PyG Data object.
            data = Data(x=types.unsqueeze(-1), pos=pos, edge_index=edge_index, y=y)

            if lattice is not None:
                data.lattice = lattice
                # Per-graph box lengths stored as (1, 3) so PyG batching collates
                # them to (B, 3). The raw lattice (3, 3) matrix does NOT collate
                # cleanly (PyG concatenates along dim 0 -> (B*3, 3)), and a
                # (3,) vector collates to (B*3,) — the diffusion module uses
                # `box` for cell conditioning and PBC wrapping.
                data.box = torch.diagonal(lattice).reshape(1, 3).clone()

            data.mol_name = mol_key
            data.frame = frame if frame is not None else 0

            logger.debug(
                "sample idx=%d mol=%s: %d nodes, %d edges",
                idx,
                mol_key,
                data.num_nodes,
                data.edge_index.shape[1],
            )
            return data


def get_h5_dataset(
    h5_path: str,
    target_key: str,
    radius: float = 6.0,
    box_key: str = "lattice",
) -> H5MolecularDataset:
    """
    Helper function to instantiate the H5MolecularDataset.
    """
    return H5MolecularDataset(
        h5_path=h5_path,
        target_key=target_key,
        radius=radius,
        box_key=box_key,
    )


class CombinedH5MolecularDataset(Dataset):
    """A PyTorch Geometric Dataset spanning several HDF5 files.

    Combines one :class:`H5MolecularDataset` per file into a single flat dataset,
    so samples behave exactly like the single-file case: each returns a ``Data``
    with ``x`` (atom types), ``pos``, ``edge_index``, ``y``, ``lattice``,
    ``mol_name`` and ``frame``. The per-file graph construction (periodic radius
    graph, CUDA fallback, ...) is fully reused.

    Args:
        h5_paths: A single path or a list of paths to HDF5 files sharing the same
            layout and target key.
        target_key: Property key(s) within each group — a string or a list to
            train on several properties at once (e.g. 'Positive VIP' or
            ['Positive VIP', 'HOMO']).
        radius: Radius for radius_graph construction.
        box_key: Key of the periodic box lattice within each group.
    """

    def __init__(
        self,
        h5_paths: str | list[str],
        target_key: str | list[str],
        radius: float = 6.0,
        box_key: str = "lattice",
    ) -> None:
        if isinstance(h5_paths, str):
            h5_paths = [h5_paths]
        self.h5_paths = list(h5_paths)
        if not self.h5_paths:
            raise ValueError("h5_paths must contain at least one file")

        self.target_key = target_key
        self.radius = radius
        self.box_key = box_key

        # One sub-dataset per file; reuses all per-file loading logic so the
        # combined dataset behaves "in the same way" as the original one.
        self.datasets = [
            H5MolecularDataset(
                path, target_key=target_key, radius=radius, box_key=box_key
            )
            for path in self.h5_paths
        ]
        # Flat sample index -> (dataset index, sample index within it).
        self._mapping = [
            (dataset_idx, sample_idx)
            for dataset_idx, ds in enumerate(self.datasets)
            for sample_idx in range(len(ds))
        ]
        self._num_samples = len(self._mapping)
        super().__init__(root=None)
        logger.info(
            "CombinedH5MolecularDataset: %d file(s) -> %d sample(s); per-file=%s",
            len(self.datasets),
            self._num_samples,
            self.file_counts(),
        )

    def __len__(self) -> int:
        return self._num_samples

    def mol_ids(self) -> list[str]:
        """Molecule identifier per sample, prefixed with the file index so the
        same molecule key in different HDF5 files stays distinct (used for
        group-aware cross-validation).
        """
        ids: list[str] = []
        for dataset_idx, ds in enumerate(self.datasets):
            ids.extend(f"{dataset_idx}:{mol}" for mol in ds.mol_ids())
        return ids

    def _target_values(self, indices=None) -> torch.Tensor:
        """Stack the target vector for the given flat sample indices: ``(n, num_targets)``."""
        if indices is None:
            indices = range(len(self))
        by_ds: dict[int, list[int]] = {}
        order: list[tuple[int, int]] = []
        for i in indices:
            di, si = self._mapping[i]
            by_ds.setdefault(di, []).append(si)
            order.append((di, len(by_ds[di]) - 1))
        pieces = {
            di: self.datasets[di]._target_values(sis) for di, sis in by_ds.items()
        }
        return torch.stack([pieces[di][pos] for di, pos in order], dim=0)

    def target_mean_std(self, indices=None):
        """Per-target mean/std over the (optionally subset) flat samples.

        Returns ``(mean, std)`` tensors of shape ``(num_targets,)``; zero-
        variance columns get ``std = 1`` so standardization stays well-defined.
        """
        y = self._target_values(indices)
        mean = y.mean(dim=0)
        std = y.std(dim=0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return mean, std

    @property
    def target_stats(self):
        """Per-target ``(mean, std)`` over all samples of this dataset."""
        return self.target_mean_std()

    def __getitem__(self, idx: int) -> Data:
        """Load the sample at flat index ``idx``, delegating to the owning file."""
        dataset_idx, sample_idx = self._mapping[idx]
        return self.datasets[dataset_idx][sample_idx]

    def file_counts(self) -> list[int]:
        """Number of samples contributed by each file, in order."""
        return [len(ds) for ds in self.datasets]


def get_combined_h5_dataset(
    h5_paths: str | list[str],
    target_key: str | list[str],
    radius: float = 6.0,
    box_key: str = "lattice",
) -> CombinedH5MolecularDataset:
    """Helper function to instantiate a CombinedH5MolecularDataset."""
    return CombinedH5MolecularDataset(
        h5_paths=h5_paths,
        target_key=target_key,
        radius=radius,
        box_key=box_key,
    )


# ---------------------------------------------------------------------------
# AMS OLED "SCM pure" HDF5 layout: one periodic box of N molecules per file.
# ---------------------------------------------------------------------------

# Element symbol -> atomic number conversion is delegated to
# :class:`morphology_gnn.periodic_table.PT` (see ``_atomic_number`` below);
# the full symbol -> atomic-number table lives only in ``periodic_table.py``.


def _atomic_number(symbol) -> int:
    """Map an element symbol (bytes or str, e.g. ``b'C'``/``'C'``) to its atomic number."""
    if isinstance(symbol, bytes):
        symbol = symbol.decode("ascii")
    return PT.get_atomic_number(symbol.strip())


def _atomic_mass(symbol) -> float:
    """Atomic weight of an element symbol (bytes or str), via the periodic table."""
    if isinstance(symbol, bytes):
        symbol = symbol.decode("ascii")
    return float(PT.get_mass(symbol.strip().capitalize()))


def _bonds_from_scm(bonds):
    """Convert an SCM ``molecules/bonds[i]`` record into 0-based ``(E, 2)`` indices.

    SCM bond records are structured arrays with ``(atom_1, atom_2, bond_order)``
    fields using **1-based** atom indices. Numeric ``(E, 2)`` inputs (or a plain
    tensor) are passed through unchanged; ``None`` stays ``None``.
    """
    if bonds is None:
        return None
    b = np.asarray(bonds)
    if b.dtype.names is not None:
        if "atom_1" in b.dtype.names and "atom_2" in b.dtype.names:
            return torch.stack(
                [
                    torch.as_tensor(b["atom_1"], dtype=torch.long) - 1,
                    torch.as_tensor(b["atom_2"], dtype=torch.long) - 1,
                ],
                dim=1,
            )
        raise ValueError(
            f"unexpected SCM bonds fields {b.dtype.names}; expected atom_1/atom_2"
        )
    return torch.as_tensor(b, dtype=torch.long)


def molecule_center_of_mass(struct_atoms, lattice, bonds=None) -> torch.Tensor:
    """PBC-aware, mass-weighted center of mass of one SCM-pure molecule.

    Matches the values stored under ``molecules/position`` in the SCM-pure HDF5
    files: the wrapped atoms are first unwrapped (:func:`unwrap_molecule`) so
    atoms that cross the periodic boundary are brought back to a contiguous
    spatial arrangement, then the center of mass is computed using the per-element
    atomic weights (from :data:`morphology_gnn.periodic_table.PT`) and folded back
    into the cell.

    Args:
        struct_atoms: The ``molecules/atoms[i]`` structured array (fields
            ``symbol``, ``x``, ``y``, ``z``).
        lattice: The periodic box lattice -- ``(3,)`` box lengths or ``(3, 3)``
            matrix.
        bonds: Optional connectivity for the graph-based unwrap: the raw SCM
            ``molecules/bonds[i]`` structured array, or a numeric ``(E, 2)``
            (0-based) index pair tensor.

    Returns:
        The center of mass, shape ``(3,)``.
    """
    atoms = np.asarray(struct_atoms)
    xyz = np.stack([atoms["x"], atoms["y"], atoms["z"]], axis=-1).astype(np.float32)
    pos = torch.tensor(xyz, dtype=torch.float)
    masses = torch.tensor([_atomic_mass(s) for s in atoms["symbol"]], dtype=torch.float)
    return pbc_center_of_mass(
        pos,
        torch.as_tensor(lattice, dtype=torch.float),
        masses=masses,
        bonds=_bonds_from_scm(bonds),
    )


# Per-molecule property groups whose datasets can be used as targets and whose
# short names (e.g. "HOMO") are resolved to full "group/dataset" paths.
_SCM_TARGET_GROUPS = (
    "energies",
    "exciton_energies",
    "static_multipole_moments",
    "transition_dipole_moments",
)


class SCMMolecularDataset(Dataset):
    """Per-molecule PyG dataset for the AMS OLED "SCM pure" HDF5 layout.

    Each file describes one periodic box containing N molecules (stored as
    per-molecule records, not per-frame trajectories). Every molecule becomes
    one sample: its atoms (``molecules/atoms``) are converted to atomic numbers
    and connected into a periodic radius graph (using ``molecules/lattice``),
    and its per-molecule properties are attached to the returned ``Data``
    object -- including the center of mass (``molecules/position``), the
    orientation, the static dipole moment and the transition dipole moments.

    Targets are selected with ``target_key``: a short name (``"HOMO"``,
    ``"S1"``, ``"dipole_moment"``) is resolved against the known per-molecule
    groups, or a full path (``"energies/HOMO"``) is used verbatim. Multi-target
    training works like :class:`H5MolecularDataset` (``y`` stacks the flattened
    per-molecule values).
    """

    def __init__(
        self,
        h5_path: str,
        target_key: str | list[str],
        radius: float = 6.0,
        box_key: str = "lattice",
    ):
        """
        Args:
            h5_path: Path to the HDF5 file (SCM pure layout).
            target_key: Property key(s) per molecule -- a string or a list,
                either a short name (e.g. 'HOMO', 'S1', 'dipole_moment') or a
                full path (e.g. 'energies/HOMO', 'transition_dipole_moments/S1_S0').
            radius: Radius for the (periodic) radius-graph construction.
            box_key: Key of the periodic box lattice within the file.
        """
        self.h5_path = h5_path
        self.target_keys = (
            [target_key] if isinstance(target_key, str) else list(target_key)
        )
        if not self.target_keys:
            raise ValueError("target_key must be a non-empty string or list of strings")
        self.radius = radius
        self.box_key = box_key

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found at: {h5_path}")

        with h5py.File(self.h5_path, "r") as hf:
            if "molecules" not in hf or "molecules/atoms" not in hf:
                raise ValueError(
                    f"{self.h5_path} is not an SCM pure layout: missing "
                    f"'molecules/atoms' (found groups: {sorted(hf.keys())})"
                )
            self.target_paths = self._resolve_target_paths(hf, self.target_keys)
            self._num_samples = len(hf["molecules/atoms"])
            self._species_names = self._read_strings(hf, "species/name")
            self._species_smiles = self._read_strings(hf, "species/smiles")
        logger.info(
            "SCMMolecularDataset %s: %d molecule(s), radius=%.3f, targets=%r -> %s",
            h5_path,
            self._num_samples,
            radius,
            self.target_keys,
            self.target_paths,
        )
        super().__init__(root=None)

    @staticmethod
    def _read_strings(hf, path) -> list[str]:
        """Read a bytes/str dataset into a list of strings (empty if absent)."""
        if path not in hf:
            return []
        raw = hf[path][:]
        return [s.decode() if isinstance(s, bytes) else str(s) for s in raw]

    def _resolve_target_paths(self, hf, target_keys) -> list[str]:
        """Map short names (``HOMO``) to ``group/dataset`` paths and validate."""
        paths = []
        for tk in target_keys:
            if "/" in tk:
                if tk not in hf:
                    raise KeyError(
                        f"target key {tk!r} not found in {self.h5_path}; "
                        f"known groups: {sorted(_SCM_TARGET_GROUPS)}"
                    )
                paths.append(tk)
            else:
                found = [f"{g}/{tk}" for g in _SCM_TARGET_GROUPS if f"{g}/{tk}" in hf]
                if not found:
                    raise KeyError(
                        f"target key {tk!r} not found in {self.h5_path}; short "
                        f"names resolve within {sorted(_SCM_TARGET_GROUPS)}"
                    )
                if len(found) > 1:
                    raise KeyError(f"target key {tk!r} is ambiguous: {found}")
                paths.append(found[0])
        return paths

    def __len__(self) -> int:
        return self._num_samples

    def mol_ids(self) -> list[str]:
        """Molecule identifier per sample: each molecule is its own group."""
        return [str(i) for i in range(self._num_samples)]

    def _target_values(self, indices=None) -> torch.Tensor:
        """Stack the target vector for the given sample indices: ``(n, num_targets)``."""
        if indices is None:
            indices = range(len(self))
        indices = list(indices)
        with h5py.File(self.h5_path, "r") as hf:
            arrays = [hf[p][:] for p in self.target_paths]
        rows = []
        for i in indices:
            vals = []
            for arr in arrays:
                vals.extend(float(v) for v in np.asarray(arr[i]).reshape(-1))
            rows.append(vals)
        return torch.tensor(rows, dtype=torch.float)

    def target_mean_std(self, indices=None):
        """Per-target mean/std over the (optionally subset) samples.

        Returns ``(mean, std)`` tensors of shape ``(num_targets,)``; zero-
        variance columns get ``std = 1`` so standardization stays well-defined.
        """
        y = self._target_values(indices)
        mean = y.mean(dim=0)
        std = y.std(dim=0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return mean, std

    @property
    def target_stats(self):
        """Per-target ``(mean, std)`` over all molecules of this dataset."""
        return self.target_mean_std()

    @property
    def lattice(self) -> torch.Tensor:
        """The periodic box lattice ``(3, 3)``."""
        with h5py.File(self.h5_path, "r") as hf:
            return torch.tensor(hf["molecules/lattice"][:], dtype=torch.float)

    def pairs(self) -> torch.Tensor:
        """Intermolecular pair indices ``(P, 2)`` (pairs within the cut-off radius)."""
        with h5py.File(self.h5_path, "r") as hf:
            return torch.tensor(hf["pairs/indices"][:], dtype=torch.long)

    def transfer_integrals(self) -> dict[str, torch.Tensor]:
        """Intermolecular transfer integrals, keyed by carrier (``electron``/``hole``)."""
        out = {}
        with h5py.File(self.h5_path, "r") as hf:
            for carrier in ("electron", "hole"):
                path = f"transfer_integrals/{carrier}"
                if path in hf:
                    out[carrier] = torch.tensor(hf[path][:], dtype=torch.float)
        return out

    def __getitem__(self, idx: int) -> Data:
        """Load one molecule's graph + its per-molecule properties."""
        with h5py.File(self.h5_path, "r") as hf:
            group = hf["molecules"]
            atoms = np.asarray(group["atoms"][idx])  # structured (symbol, x, y, z)
            xyz = np.stack([atoms["x"], atoms["y"], atoms["z"]], axis=-1).astype(
                np.float32
            )
            pos = torch.tensor(xyz, dtype=torch.float)
            types = torch.tensor(
                [_atomic_number(s) for s in atoms["symbol"]], dtype=torch.long
            )

            ys = []
            for p in self.target_paths:
                ys.append(
                    torch.tensor(np.asarray(hf[p][idx]).reshape(-1), dtype=torch.float)
                )
            y = torch.cat(ys)

            lattice = torch.tensor(hf["molecules/lattice"][:], dtype=torch.float)

        # Periodic radius graph over the molecule's atoms (in the box frame).
        if cuda_radius_graph_pbc is not None and _cuda_available:
            try:
                edge_index = cuda_radius_graph_pbc(
                    pos, r=self.radius, lattice=lattice, loop=False
                )
            except Exception:
                edge_index = radius_graph_pbc(
                    pos, r=self.radius, lattice=lattice, loop=False
                )
        else:
            edge_index = radius_graph_pbc(
                pos, r=self.radius, lattice=lattice, loop=False
            )

        data = Data(x=types.unsqueeze(-1), pos=pos, edge_index=edge_index, y=y)
        data.lattice = lattice
        data.box = torch.diagonal(lattice).reshape(1, 3).clone()
        data.mol_name = str(idx)
        data.n_atoms = int(len(atoms))

        # --- extra per-molecule data (attached when present in the file) ------
        # Vector quantities are stored as (1, 3) so PyG batching collates them
        # to (B, 3) -- the same convention as `box` above; a bare (3,) vector
        # would collate to (B*3,).
        with h5py.File(self.h5_path, "r") as hf:
            group = hf["molecules"]
            if "position" in group:
                data.com = torch.tensor(
                    group["position"][idx], dtype=torch.float
                ).reshape(1, 3)
            else:
                data.com = pos.mean(dim=0, keepdim=True)  # fallback: atom centroid
            if "orientation" in group:
                data.orientation = torch.tensor(
                    group["orientation"][idx], dtype=torch.float
                ).reshape(1, 3)
            data.species = int(group["species"][idx])
            if "static_multipole_moments/dipole_moment" in hf:
                data.dipole_moment = torch.tensor(
                    hf["static_multipole_moments/dipole_moment"][idx], dtype=torch.float
                ).reshape(1, 3)
            for field in ("S1_S0", "T1_S0"):
                path = f"transition_dipole_moments/{field}"
                if path in hf:
                    data[f"transition_dipole_{field}"] = torch.tensor(
                        hf[path][idx], dtype=torch.float
                    ).reshape(1, 3)
        if self._species_names:
            data.species_name = self._species_names[data.species]
        if self._species_smiles:
            data.species_smiles = self._species_smiles[data.species]

        logger.debug(
            "SCM sample idx=%d: %d atoms, %d edges, com=%s",
            idx,
            data.n_atoms,
            data.edge_index.shape[1],
            data.com.tolist(),
        )
        return data


class CombinedSCMMolecularDataset(Dataset):
    """A PyTorch Geometric Dataset spanning several SCM-pure HDF5 files.

    Combines one :class:`SCMMolecularDataset` per file into a single flat
    dataset, so samples behave exactly like the single-file case: each returns
    a ``Data`` with ``x`` (atom types), ``pos``, ``edge_index``, ``y``,
    ``com`` (center of mass), ``lattice``, ``box``, ``mol_name`` and the
    per-molecule properties (orientation, dipoles, species, ...).

    Args:
        h5_paths: A single path or a list of paths to SCM-pure HDF5 files.
        target_key: Property key(s) per molecule -- a string or a list (short
            name or full ``group/dataset`` path).
        radius: Radius for the periodic radius-graph construction.
        box_key: Key of the periodic box lattice within each file.
    """

    def __init__(
        self,
        h5_paths: str | list[str],
        target_key: str | list[str],
        radius: float = 6.0,
        box_key: str = "lattice",
    ) -> None:
        if isinstance(h5_paths, str):
            h5_paths = [h5_paths]
        self.h5_paths = list(h5_paths)
        if not self.h5_paths:
            raise ValueError("h5_paths must contain at least one file")

        self.target_key = target_key
        self.radius = radius
        self.box_key = box_key

        # One sub-dataset per file; reuses all per-file loading logic so the
        # combined dataset behaves "in the same way" as the original one.
        self.datasets = [
            SCMMolecularDataset(
                path, target_key=target_key, radius=radius, box_key=box_key
            )
            for path in self.h5_paths
        ]
        # Flat sample index -> (dataset index, sample index within it).
        self._mapping = [
            (dataset_idx, sample_idx)
            for dataset_idx, ds in enumerate(self.datasets)
            for sample_idx in range(len(ds))
        ]
        self._num_samples = len(self._mapping)
        super().__init__(root=None)
        logger.info(
            "CombinedSCMMolecularDataset: %d file(s) -> %d sample(s); per-file=%s",
            len(self.datasets),
            self._num_samples,
            self.file_counts(),
        )

    def __len__(self) -> int:
        return self._num_samples

    def mol_ids(self) -> list[str]:
        """Molecule identifier per sample, prefixed with the file index so the
        same molecule index in different HDF5 files stays distinct.
        """
        ids: list[str] = []
        for dataset_idx, ds in enumerate(self.datasets):
            ids.extend(f"{dataset_idx}:{mol}" for mol in ds.mol_ids())
        return ids

    def _target_values(self, indices=None) -> torch.Tensor:
        """Stack the target vector for the given flat sample indices: ``(n, num_targets)``."""
        if indices is None:
            indices = range(len(self))
        by_ds: dict[int, list[int]] = {}
        order: list[tuple[int, int]] = []
        for i in indices:
            di, si = self._mapping[i]
            by_ds.setdefault(di, []).append(si)
            order.append((di, len(by_ds[di]) - 1))
        pieces = {
            di: self.datasets[di]._target_values(sis) for di, sis in by_ds.items()
        }
        return torch.stack([pieces[di][pos] for di, pos in order], dim=0)

    def target_mean_std(self, indices=None):
        """Per-target mean/std over the (optionally subset) flat samples."""
        y = self._target_values(indices)
        mean = y.mean(dim=0)
        std = y.std(dim=0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return mean, std

    @property
    def target_stats(self):
        """Per-target ``(mean, std)`` over all samples of this dataset."""
        return self.target_mean_std()

    def __getitem__(self, idx: int) -> Data:
        """Load the sample at flat index ``idx``, delegating to the owning file."""
        dataset_idx, sample_idx = self._mapping[idx]
        return self.datasets[dataset_idx][sample_idx]

    def file_counts(self) -> list[int]:
        """Number of samples contributed by each file, in order."""
        return [len(ds) for ds in self.datasets]


def get_scm_dataset(
    h5_path: str,
    target_key: str | list[str],
    radius: float = 6.0,
    box_key: str = "lattice",
) -> SCMMolecularDataset:
    """Helper function to instantiate an SCMMolecularDataset."""
    return SCMMolecularDataset(
        h5_path=h5_path,
        target_key=target_key,
        radius=radius,
        box_key=box_key,
    )


def get_combined_scm_dataset(
    h5_paths: str | list[str],
    target_key: str | list[str],
    radius: float = 6.0,
    box_key: str = "lattice",
) -> CombinedSCMMolecularDataset:
    """Helper function to instantiate a CombinedSCMMolecularDataset."""
    return CombinedSCMMolecularDataset(
        h5_paths=h5_paths,
        target_key=target_key,
        radius=radius,
        box_key=box_key,
    )


# ---------------------------------------------------------------------------
# Diffusion-specific SCM "box" dataset: one sample = one periodic box; the
# diffused coordinates are the molecule center-of-mass positions.
# ---------------------------------------------------------------------------


class SCMDiffusionDataset(Dataset):
    """One PyG sample per SCM-pure periodic box, for the diffusion position-generator.

    Each SCM-pure HDF5 file describes one periodic box of N molecules. This
    dataset exposes the box as a single sample whose *nodes are the molecules*:
    ``pos`` holds the molecule center-of-mass positions (``molecules/position``),
    ``x`` the per-molecule species index, and ``edge_index`` a periodic radius
    graph over the centers of mass. The diffusion model learns to denoise /
    generate these COM coordinates (i.e. the molecular packing of the box).

    The per-molecule reference data needed for generation evaluation and
    visualization (full atom records, orientation, species names) is *not*
    attached to the returned ``Data`` (object arrays do not collate under PyG
    batching) — it is exposed through :meth:`box_reference`.
    """

    def __init__(
        self,
        h5_path: str,
        target_key: str | list[str] | None = None,
        radius: float = 20.0,
        box_key: str = "lattice",
    ) -> None:
        self.h5_path = h5_path
        self.target_key = target_key
        self.radius = radius
        self.box_key = box_key

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found at: {h5_path}")

        with h5py.File(self.h5_path, "r") as hf:
            if "molecules" not in hf or "molecules/atoms" not in hf:
                raise ValueError(
                    f"{self.h5_path} is not an SCM pure layout: missing "
                    f"'molecules/atoms' (found groups: {sorted(hf.keys())})"
                )
            self._n_molecules = int(len(hf["molecules/atoms"]))
            lattice = torch.tensor(hf["molecules/lattice"][:], dtype=torch.float)
            if target_key is not None:
                self.target_paths = self._resolve_target_paths(hf, [target_key])
            else:
                self.target_paths = []
            self._species_names = self._read_strings(hf, "species/name")
            self._species_smiles = self._read_strings(hf, "species/smiles")

        self.lattice = lattice
        self.box, self.is_orthorhombic = _normalize_lattice(lattice)
        self.mol_name = os.path.splitext(os.path.basename(h5_path))[0]
        logger.info(
            "SCMDiffusionDataset %s: %d molecule(s) in one box, radius=%.3f, "
            "orthorhombic=%s",
            h5_path,
            self._n_molecules,
            radius,
            self.is_orthorhombic,
        )
        super().__init__(root=None)

    @staticmethod
    def _read_strings(hf, path) -> list[str]:
        """Read a bytes/str dataset into a list of strings (empty if absent)."""
        if path not in hf:
            return []
        raw = hf[path][:]
        return [s.decode() if isinstance(s, bytes) else str(s) for s in raw]

    def _resolve_target_paths(self, hf, target_keys) -> list[str]:
        """Map short target names (``HOMO``) to ``group/dataset`` paths and validate."""
        paths = []
        for tk in target_keys:
            if "/" in tk:
                if tk not in hf:
                    raise KeyError(f"target key {tk!r} not found in {self.h5_path}")
                paths.append(tk)
            else:
                found = [f"{g}/{tk}" for g in _SCM_TARGET_GROUPS if f"{g}/{tk}" in hf]
                if not found:
                    raise KeyError(
                        f"target key {tk!r} not found in {self.h5_path}; short "
                        f"names resolve within {sorted(_SCM_TARGET_GROUPS)}"
                    )
                if len(found) > 1:
                    raise KeyError(f"target key {tk!r} is ambiguous: {found}")
                paths.append(found[0])
        return paths

    def __len__(self) -> int:
        return 1  # one box per file

    def mol_ids(self) -> list[str]:
        """Molecule identifier per sample: one box per file."""
        return [self.mol_name]

    def __getitem__(self, idx: int) -> Data:
        with h5py.File(self.h5_path, "r") as hf:
            group = hf["molecules"]
            if "position" in group:
                pos = torch.tensor(
                    group["position"][:], dtype=torch.float
                )  # COMs (N, 3)
            else:
                # Fallback: recompute the PBC-aware COM from the raw atoms.
                coms = [
                    molecule_center_of_mass(
                        np.asarray(group["atoms"][i]), self.lattice.numpy()
                    )
                    for i in range(self._n_molecules)
                ]
                pos = torch.stack(coms)
            species = torch.tensor(group["species"][:], dtype=torch.long).reshape(-1, 1)

        edge_index = radius_graph_pbc(
            pos, r=self.radius, lattice=self.lattice, loop=False
        )
        data = Data(x=species, pos=pos, edge_index=edge_index, y=torch.zeros(1))
        data.lattice = self.lattice
        data.box = self.box.reshape(1, 3).clone()
        data.is_orthorhombic = torch.tensor(
            [int(self.is_orthorhombic)], dtype=torch.long
        )
        data.mol_name = self.mol_name
        return data

    def box_reference(self, idx: int = 0) -> dict:
        """Per-molecule reference metadata for one box (generation eval / viz).

        Returns a plain dict (not a ``Data``) holding the box lattice, the stored
        center-of-mass positions, per-molecule orientations, species, and the raw
        ``molecules/atoms`` records — enough to reconstruct full molecular
        conformations at generated COM positions.
        """
        with h5py.File(self.h5_path, "r") as hf:
            group = hf["molecules"]
            n = int(len(group["atoms"]))
            atoms = [np.asarray(group["atoms"][i]) for i in range(n)]
            orientation = (
                torch.tensor(group["orientation"][:], dtype=torch.float)
                if "orientation" in group
                else None
            )
            com = (
                torch.tensor(group["position"][:], dtype=torch.float)
                if "position" in group
                else None
            )
        return {
            "mol_name": self.mol_name,
            "lattice": self.lattice.clone(),
            "box": self.box.reshape(1, 3).clone(),
            "com": com,
            "orientation": orientation,
            "atoms": atoms,
            "species_names": self._species_names,
            "species_smiles": self._species_smiles,
        }


class CombinedSCMDiffusionDataset(Dataset):
    """A PyTorch Geometric Dataset spanning several SCM-pure boxes.

    Each file contributes one sample (one box of N molecule COM positions), so
    the combined dataset exposes ``len(h5_paths)`` samples. Samples behave like
    the single-file case: ``x`` (species), ``pos`` (COM), ``edge_index`` (PBC
    radius graph over COMs), ``box``, ``lattice``, ``mol_name``.
    """

    def __init__(
        self,
        h5_paths: str | list[str],
        target_key: str | list[str] | None = None,
        radius: float = 20.0,
        box_key: str = "lattice",
    ) -> None:
        if isinstance(h5_paths, str):
            h5_paths = [h5_paths]
        self.h5_paths = list(h5_paths)
        if not self.h5_paths:
            raise ValueError("h5_paths must contain at least one file")

        self.target_key = target_key
        self.radius = radius
        self.box_key = box_key
        self.datasets = [
            SCMDiffusionDataset(
                path, target_key=target_key, radius=radius, box_key=box_key
            )
            for path in self.h5_paths
        ]
        # One sample per file.
        self._mapping = [(di, 0) for di in range(len(self.datasets))]
        self._num_samples = len(self._mapping)
        super().__init__(root=None)
        logger.info(
            "CombinedSCMDiffusionDataset: %d file(s) -> %d box sample(s)",
            len(self.datasets),
            self._num_samples,
        )

    def __len__(self) -> int:
        return self._num_samples

    def mol_ids(self) -> list[str]:
        """Molecule identifier per sample: one box per file, file-qualified."""
        return [
            f"{di}:{mol}" for di, ds in enumerate(self.datasets) for mol in ds.mol_ids()
        ]

    def __getitem__(self, idx: int) -> Data:
        """Load the box at flat index ``idx``, delegating to the owning file."""
        dataset_idx, sample_idx = self._mapping[idx]
        return self.datasets[dataset_idx][sample_idx]

    def box_reference(self, idx: int = 0) -> dict:
        """Per-molecule reference metadata for the box at flat index ``idx``."""
        dataset_idx, sample_idx = self._mapping[idx]
        return self.datasets[dataset_idx].box_reference(sample_idx)

    def file_counts(self) -> list[int]:
        """Number of box samples contributed by each file, in order."""
        return [1 for _ in self.datasets]


def get_scm_diffusion_dataset(
    h5_path: str,
    target_key: str | list[str] | None = None,
    radius: float = 20.0,
    box_key: str = "lattice",
) -> SCMDiffusionDataset:
    """Helper function to instantiate an SCMDiffusionDataset."""
    return SCMDiffusionDataset(
        h5_path, target_key=target_key, radius=radius, box_key=box_key
    )


def get_combined_scm_diffusion_dataset(
    h5_paths: str | list[str],
    target_key: str | list[str] | None = None,
    radius: float = 20.0,
    box_key: str = "lattice",
) -> CombinedSCMDiffusionDataset:
    """Helper function to instantiate a CombinedSCMDiffusionDataset."""
    return CombinedSCMDiffusionDataset(
        h5_paths, target_key=target_key, radius=radius, box_key=box_key
    )
