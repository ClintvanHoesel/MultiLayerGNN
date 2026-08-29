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
    min_image_disp,
    pbc_center_of_mass,
    radius_graph_pbc,
    unwrap_molecule,
    try_cuda_radius_graph_pbc,
)

logger = logging.getLogger(__name__)


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
                edge_index = try_cuda_radius_graph_pbc(
                    pos,
                    r=self.radius,
                    lattice=lattice,
                    loop=False,
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
# AMS OLED "box" HDF5 layout: one periodic box of N molecules per file.
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


def _bonds_from_box(bonds):
    """Convert a box ``molecules/bonds[i]`` record into 0-based ``(E, 2)`` indices.

    Box bond records are structured arrays with ``(atom_1, atom_2, bond_order)``
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
            f"unexpected box bonds fields {b.dtype.names}; expected atom_1/atom_2"
        )
    return torch.as_tensor(b, dtype=torch.long)


def molecule_center_of_mass(struct_atoms, lattice, bonds=None) -> torch.Tensor:
    """PBC-aware, mass-weighted center of mass of one box molecule.

    Matches the values stored under ``molecules/position`` in the box HDF5
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
        bonds: Optional connectivity for the graph-based unwrap: the raw box
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
        bonds=_bonds_from_box(bonds),
    )


# Per-molecule property groups whose datasets can be used as targets and whose
# short names (e.g. "HOMO") are resolved to full "group/dataset" paths.
_BOX_TARGET_GROUPS = (
    "energies",
    "exciton_energies",
    "static_multipole_moments",
    "transition_dipole_moments",
)


class BoxMolecularDataset(Dataset):
    """Per-molecule PyG dataset for the AMS OLED "box" HDF5 layout.

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
        target_key: str | list[str] | None = None,
        radius: float = 6.0,
        box_key: str = "lattice",
        keep_in_memory: bool = False,
        context: dict | None = None,
    ):
        """
        Args:
            h5_path: Path to the HDF5 file (box layout).
            target_key: Property key(s) per molecule -- a string, a list, or
                ``None`` when only box-level access is needed (no targets). A
                short name (e.g. 'HOMO', 'S1', 'dipole_moment') or a full path
                (e.g. 'energies/HOMO', 'transition_dipole_moments/S1_S0').
            radius: Radius for the (periodic) radius-graph construction.
            box_key: Key of the periodic box lattice within the file.
            keep_in_memory: When True, every molecule's data (atoms, targets,
                COMs, orientations, ...) is loaded into memory once at
                construction so ``__getitem__`` and the other accessors never
                re-open / re-read the HDF5 file.
            context: Optional per-molecule context configuration
                ``{"mode": "radius"|"knn"|"all", "radius": 20.0, "k": 8}``.
                When set, every sample's graph additionally contains the atoms
                of the query molecule's surrounding molecules (radius
                neighbours / k nearest / the whole box). Node-level
                ``mol_number`` (query = 0) and ``mol_is_query`` (bool) mark the
                query vs context atoms, while ``y`` and the per-molecule
                metadata always describe the query molecule, so only the query
                molecule of each sample is ever trained on.
        """
        self.h5_path = h5_path
        self.target_keys = (
            [target_key]
            if isinstance(target_key, str)
            else list(target_key) if target_key is not None else []
        )
        self.radius = radius
        self.box_key = box_key
        self.keep_in_memory = bool(keep_in_memory)
        self.context = dict(context or {})
        self._has_context = bool(self.context)
        self._cache: dict | None = None
        self._atoms_cache: list | None = None
        self._context_neighbors: list[list[int]] | None = None

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found at: {h5_path}")

        with h5py.File(self.h5_path, "r") as hf:
            if "molecules" not in hf or "molecules/atoms" not in hf:
                raise ValueError(
                    f"{self.h5_path} is not a box layout: missing "
                    f"'molecules/atoms' (found groups: {sorted(hf.keys())})"
                )
            self.target_paths = (
                self._resolve_target_paths(hf, self.target_keys)
                if self.target_keys
                else []
            )
            self._num_samples = len(hf["molecules/atoms"])
            self._species_names = self._read_strings(hf, "species/name")
            self._species_smiles = self._read_strings(hf, "species/smiles")
        if self.keep_in_memory:
            self._cache = self._build_cache()
        if self._has_context:
            self._context_neighbors = self._build_context_neighbors()
            logger.info(
                "BoxMolecularDataset %s: context mode=%r (radius=%.1f, k=%d) "
                "neighbours precomputed for %d molecule(s)",
                h5_path,
                self.context.get("mode", "radius"),
                self.context.get("radius", 20.0),
                self.context.get("k", 8),
                self._num_samples,
            )
        logger.info(
            "BoxMolecularDataset %s: %d molecule(s), radius=%.3f, targets=%r -> %s, "
            "in_memory=%s, context=%s",
            h5_path,
            self._num_samples,
            radius,
            self.target_keys,
            self.target_paths,
            self.keep_in_memory,
            bool(self._has_context),
        )
        super().__init__(root=None)

    def _build_cache(self) -> dict:
        """Load every molecule's data from the HDF5 file into memory once.

        Only called when ``keep_in_memory=True``; all accessors then read from
        this dict instead of re-opening the file on every call.
        """
        cache: dict = {}
        with h5py.File(self.h5_path, "r") as hf:
            group = hf["molecules"]
            n = int(len(group["atoms"]))
            cache["n"] = n
            cache["atoms"] = [np.asarray(group["atoms"][i]) for i in range(n)]
            cache["lattice"] = torch.tensor(
                hf["molecules/lattice"][:], dtype=torch.float
            )
            cache["position"] = (
                torch.tensor(group["position"][:], dtype=torch.float)
                if "position" in group
                else None
            )
            cache["orientation"] = (
                torch.tensor(group["orientation"][:], dtype=torch.float)
                if "orientation" in group
                else None
            )
            cache["species"] = torch.tensor(group["species"][:], dtype=torch.long)
            cache["targets"] = {
                p: torch.tensor(hf[p][:], dtype=torch.float) for p in self.target_paths
            }
            cache["dipole_moment"] = (
                torch.tensor(
                    hf["static_multipole_moments/dipole_moment"][:], dtype=torch.float
                )
                if "static_multipole_moments/dipole_moment" in hf
                else None
            )
            for field in ("S1_S0", "T1_S0"):
                path = f"transition_dipole_moments/{field}"
                cache[f"transition_dipole_{field}"] = (
                    torch.tensor(hf[path][:], dtype=torch.float) if path in hf else None
                )
            cache["pairs"] = (
                torch.tensor(hf["pairs/indices"][:], dtype=torch.long)
                if "pairs/indices" in hf
                else None
            )
            cache["energies"] = {
                energy: torch.tensor(hf[f"energies/{energy}"][:], dtype=torch.float)
                for energy in ("IP", "EA", "HOMO", "LUMO", "HOMO-1", "LUMO+1")
                if f"energies/{energy}" in hf
            }
            cache["transfer_integrals"] = {
                carrier: torch.tensor(
                    hf[f"transfer_integrals/{carrier}"][:], dtype=torch.float
                )
                for carrier in ("electron", "hole")
                if f"transfer_integrals/{carrier}" in hf
            }
        return cache

    def _row(self, idx: int) -> dict:
        """All per-molecule data for molecule ``idx`` (cache or HDF5).

        The single place that touches the HDF5 file for per-molecule samples:
        it reads ``molecules/atoms``, the lattice, the targets, and every
        optional per-molecule quantity (COM, orientation, species, dipoles).
        When ``keep_in_memory`` is set the values come straight from
        ``self._cache``.
        """
        if self._cache is not None:
            c = self._cache
            return {
                "atoms": c["atoms"][idx],
                "lattice": c["lattice"],
                "position": None if c["position"] is None else c["position"][idx],
                "orientation": (
                    None if c["orientation"] is None else c["orientation"][idx]
                ),
                "species": int(c["species"][idx]),
                "targets": [
                    c["targets"][p][idx].reshape(-1) for p in self.target_paths
                ],
                "energies": [c["energies"][p][idx].reshape(-1) for p in c["energies"]],
                "dipole_moment": (
                    None if c["dipole_moment"] is None else c["dipole_moment"][idx]
                ),
                "transition_dipole_S1_S0": (
                    None
                    if c["transition_dipole_S1_S0"] is None
                    else c["transition_dipole_S1_S0"][idx]
                ),
                "transition_dipole_T1_S0": (
                    None
                    if c["transition_dipole_T1_S0"] is None
                    else c["transition_dipole_T1_S0"][idx]
                ),
            }
        with h5py.File(self.h5_path, "r") as hf:
            group = hf["molecules"]
            atoms = np.asarray(group["atoms"][idx])
            lattice = torch.tensor(hf["molecules/lattice"][:], dtype=torch.float)
            row: dict = {
                "atoms": atoms,
                "lattice": lattice,
                "position": (
                    torch.tensor(group["position"][idx], dtype=torch.float)
                    if "position" in group
                    else None
                ),
                "orientation": (
                    torch.tensor(group["orientation"][idx], dtype=torch.float)
                    if "orientation" in group
                    else None
                ),
                "species": int(group["species"][idx]),
                "targets": [
                    torch.tensor(np.asarray(hf[p][idx]).reshape(-1), dtype=torch.float)
                    for p in self.target_paths
                ],
                "dipole_moment": (
                    torch.tensor(
                        hf["static_multipole_moments/dipole_moment"][idx],
                        dtype=torch.float,
                    )
                    if "static_multipole_moments/dipole_moment" in hf
                    else None
                ),
            }
            for field in ("IP", "EA", "HOMO", "LUMO", "HOMO-1", "LUMO-1"):
                path = f"energy/{field}"
                row[f"energy_{field}"] = (
                    torch.tensor(hf[path][idx], dtype=torch.float)
                    if path in hf
                    else None
                )
            for field in ("S1_S0", "T1_S0"):
                path = f"transition_dipole_moments/{field}"
                row[f"transition_dipole_{field}"] = (
                    torch.tensor(hf[path][idx], dtype=torch.float)
                    if path in hf
                    else None
                )
        return row

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
                        f"known groups: {sorted(_BOX_TARGET_GROUPS)}"
                    )
                paths.append(tk)
            else:
                found = [f"{g}/{tk}" for g in _BOX_TARGET_GROUPS if f"{g}/{tk}" in hf]
                if not found:
                    raise KeyError(
                        f"target key {tk!r} not found in {self.h5_path}; short "
                        f"names resolve within {sorted(_BOX_TARGET_GROUPS)}"
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
        if not self.target_paths:
            return torch.empty((len(indices), 0), dtype=torch.float)
        if self._cache is not None:
            arrays = [self._cache["targets"][p] for p in self.target_paths]
        else:
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
        if self._cache is not None:
            return self._cache["lattice"]
        with h5py.File(self.h5_path, "r") as hf:
            return torch.tensor(hf["molecules/lattice"][:], dtype=torch.float)

    def pairs(self) -> torch.Tensor:
        """Intermolecular pair indices ``(P, 2)`` (pairs within the cut-off radius)."""
        if self._cache is not None:
            val = self._cache["pairs"]
            if val is None:
                raise KeyError("pairs/indices not found")
            return val
        with h5py.File(self.h5_path, "r") as hf:
            return torch.tensor(hf["pairs/indices"][:], dtype=torch.long)

    def transfer_integrals(self) -> dict[str, torch.Tensor]:
        """Intermolecular transfer integrals, keyed by carrier (``electron``/``hole``)."""
        if self._cache is not None:
            return dict(self._cache["transfer_integrals"])
        out = {}
        with h5py.File(self.h5_path, "r") as hf:
            for carrier in ("electron", "hole"):
                path = f"transfer_integrals/{carrier}"
                if path in hf:
                    out[carrier] = torch.tensor(hf[path][:], dtype=torch.float)
        return out

    @property
    def mol_name(self) -> str:
        """Box (file) name: the HDF5 basename without extension.

        Per-molecule samples expose ``data.mol_name = str(idx)``; this is the
        per-*file* name used by the box-level accessors (``box_sample``,
        ``box_reference``).
        """
        return os.path.splitext(os.path.basename(self.h5_path))[0]

    def coms(self) -> torch.Tensor:
        """Molecule center-of-mass positions ``(N, 3)`` for this box.

        Uses the stored ``molecules/position`` when present, otherwise falls
        back to the PBC-aware, mass-weighted center of mass recomputed from the
        raw atoms (:func:`molecule_center_of_mass`).
        """
        if self._cache is not None:
            if self._cache["position"] is not None:
                return self._cache["position"]
            return torch.stack(
                [
                    molecule_center_of_mass(
                        self._cache["atoms"][i], self._cache["lattice"]
                    )
                    for i in range(self._cache["n"])
                ]
            )
        with h5py.File(self.h5_path, "r") as hf:
            group = hf["molecules"]
            if "position" in group:
                return torch.tensor(group["position"][:], dtype=torch.float)
            lattice = torch.tensor(hf["molecules/lattice"][:], dtype=torch.float)
            return torch.stack(
                [
                    molecule_center_of_mass(np.asarray(group["atoms"][i]), lattice)
                    for i in range(int(len(group["atoms"])))
                ]
            )

    def species_ids(self) -> torch.Tensor:
        """Per-molecule species indices ``(N,)`` for this box."""
        if self._cache is not None:
            return self._cache["species"].reshape(-1)
        with h5py.File(self.h5_path, "r") as hf:
            return torch.tensor(hf["molecules/species"][:], dtype=torch.long).reshape(
                -1
            )

    def box_sample(self, radius: float | None = None) -> Data:
        """One box-level sample: *molecules as nodes* for diffusion.

        ``pos`` holds the molecule center-of-mass positions (:meth:`coms`), ``x``
        the per-molecule species index, and ``edge_index`` a periodic radius
        graph over the COMs (``radius`` defaults to 20 Å, the COM-scale cutoff).
        Returns a ``Data`` with ``pos``, ``x``, ``edge_index``, ``box`` ``(1, 3)``,
        ``lattice``, ``is_orthorhombic`` and ``mol_name`` — the same layout the
        diffusion model consumes for molecular-packing generation.
        """
        radius = 20.0 if radius is None else radius
        pos = self.coms()
        species = self.species_ids().reshape(-1, 1)
        lattice = self.lattice
        edge_index = radius_graph_pbc(pos, r=radius, lattice=lattice, loop=False)
        box, is_orthorhombic = _normalize_lattice(lattice)
        data = Data(x=species, pos=pos, edge_index=edge_index, y=torch.zeros(1))
        data.lattice = lattice.clone()
        data.box = box.reshape(1, 3).clone()
        data.is_orthorhombic = torch.tensor([int(is_orthorhombic)], dtype=torch.long)
        data.mol_name = self.mol_name
        return data

    def box_reference(self) -> dict:
        """Per-molecule reference metadata for this box (generation eval / viz).

        Returns a plain dict holding the box lattice, stored center-of-mass
        positions, per-molecule orientations, species, and the raw
        ``molecules/atoms`` records — enough to reconstruct full molecular
        conformations at generated COM positions.
        """
        if self._cache is not None:
            assert _normalize_lattice(self._cache["lattice"])[1]
            return {
                "mol_name": self.mol_name,
                "lattice": self._cache["lattice"].clone(),
                "box": _normalize_lattice(self._cache["lattice"])[0]
                .reshape(1, 3)
                .clone(),
                "com": self._cache["position"],
                "orientation": self._cache["orientation"],
                "atoms": self._cache["atoms"],
                "species_names": self._species_names,
                "species_smiles": self._species_smiles,
            }
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
        assert _normalize_lattice(self.lattice)[1]
        return {
            "mol_name": self.mol_name,
            "lattice": self.lattice.clone(),
            "box": _normalize_lattice(self.lattice)[0].reshape(1, 3).clone(),
            "com": com,
            "orientation": orientation,
            "atoms": atoms,
            "species_names": self._species_names,
            "species_smiles": self._species_smiles,
        }

    def _atoms_list(self) -> list:
        """All molecule atom records, in memory, for building context graphs.

        Reads every ``molecules/atoms[i]`` once and caches the list (the cached
        ``keep_in_memory`` records are reused when available) so context-mode
        ``__getitem__`` does not re-open / re-read the HDF5 file per sample.
        """
        if self._cache is not None:
            return self._cache["atoms"]
        if self._atoms_cache is None:
            with h5py.File(self.h5_path, "r") as hf:
                group = hf["molecules"]
                n = int(len(group["atoms"]))
                self._atoms_cache = [np.asarray(group["atoms"][i]) for i in range(n)]
        return self._atoms_cache

    def _build_context_neighbors(self) -> list[list[int]]:
        """Per-query list of context molecule indices (PBC-aware).

        ``mode="radius"``: every molecule whose center of mass lies within
        ``context.radius`` Angstrom of the query's COM. ``mode="knn"``: the
        ``context.k`` nearest molecules by COM distance. ``mode="all"``: every
        other molecule in the box. The query molecule is never its own context.
        """
        mode = self.context.get("mode", "radius")
        n = self._num_samples
        if mode == "all":
            return [[j for j in range(n) if j != i] for i in range(n)]
        coms = self.coms()
        lattice = self.lattice
        if mode == "knn":
            k = max(int(self.context.get("k", 8)), 1)
            src = torch.arange(n).repeat_interleave(n)
            dst = torch.arange(n).repeat(n)
            keep = src != dst
            src, dst = src[keep], dst[keep]
            dists = min_image_disp(coms, torch.stack([src, dst]), lattice).norm(dim=1)
            out: list[list[int]] = []
            for i in range(n):
                cand = dst[src == i]
                order = torch.argsort(dists[src == i])[:k]
                out.append(cand[order].tolist())
            return out
        if mode == "radius":
            r = float(self.context.get("radius", 20.0))
            box = torch.diagonal(lattice).reshape(-1)
            if box.numel() == 3 and float(box.min()) < 2.0 * r:
                logger.warning(
                    "context radius %.1f is close to or exceeds half a box "
                    "length (%s); minimum-image neighbourhoods may be ambiguous",
                    r,
                    box.tolist(),
                )
            edge = radius_graph_pbc(coms, r=r, lattice=lattice, loop=False)
            out = [[] for _ in range(n)]
            for i, j in zip(edge[0].tolist(), edge[1].tolist()):
                if i != j:
                    out[i].append(j)
            return out
        raise ValueError(
            f"unknown context.mode {mode!r}; choose from 'radius', 'knn', 'all'"
        )

    def _build_context_graph(self, idx: int, row: dict):
        """Concatenate the query molecule's atoms with its context molecules'.

        Returns ``(pos, types, n_per, mol_number, mol_is_query)``:
        ``pos`` ``(N, 3)`` box-frame coordinates of every included atom,
        ``types`` ``(N,)`` atomic numbers, ``n_per`` the number of atoms per
        molecule (query first), ``mol_number`` ``(N,)`` the actual molecule id
        each node belongs to (query = ``idx``, context = the neighbour molecule
        indices) and ``mol_is_query`` a boolean mask over nodes that is True
        only for the query molecule's atoms.
        """
        neighbors = self._context_neighbors[idx]
        blocks = [np.asarray(row["atoms"])] + [self._atoms_list()[j] for j in neighbors]
        n_per = [len(a) for a in blocks]
        xyz = np.concatenate(
            [
                np.stack([a["x"], a["y"], a["z"]], axis=-1).astype(np.float32)
                for a in blocks
            ],
            axis=0,
        )
        types = torch.tensor(
            np.concatenate(
                [np.array([_atomic_number(s) for s in a["symbol"]]) for a in blocks]
            ),
            dtype=torch.long,
        )
        pos = torch.tensor(xyz, dtype=torch.float)
        # Actual molecule ids per node: the query is molecule ``idx``, each
        # context block belongs to its neighbour molecule index (the same ``j``
        # as in ``neighbors``).
        mol_ids = [idx] + list(neighbors)
        mol_number = torch.tensor(mol_ids, dtype=torch.long).repeat_interleave(
            torch.tensor(n_per, dtype=torch.long)
        )
        mol_is_query = torch.zeros(pos.shape[0], dtype=torch.bool)
        mol_is_query[: n_per[0]] = True
        return pos, types, n_per, mol_number, mol_is_query

    def __getitem__(self, idx: int) -> Data:
        """Load one molecule's graph + its per-molecule properties.

        When ``context`` is configured the returned graph also contains the
        atoms of the query molecule's surrounding molecules (radius neighbours,
        k-NN, or the whole box). Node-level ``mol_number`` (query = 0) and
        ``mol_is_query`` (bool) mark which atoms belong to the trained query
        molecule vs. the (untrained) context; ``y`` and every per-molecule
        metadata field always describe the query molecule.
        """
        row = self._row(idx)
        lattice = row["lattice"]

        if self._has_context:
            pos, types, n_per, mol_number, mol_is_query = self._build_context_graph(
                idx, row
            )
            n_query_atoms = int(n_per[0])
        else:
            atoms = np.asarray(row["atoms"])
            xyz = np.stack([atoms["x"], atoms["y"], atoms["z"]], axis=-1).astype(
                np.float32
            )
            pos = torch.tensor(xyz, dtype=torch.float)
            types = torch.tensor(
                [_atomic_number(s) for s in atoms["symbol"]], dtype=torch.long
            )
            n_query_atoms = int(len(types))
            mol_number = mol_is_query = None

        # Periodic radius graph over all included atoms (query + context), in
        # the shared box frame -- naturally produces intra- and inter-molecular
        # edges under PBC.
        edge_index = try_cuda_radius_graph_pbc(
            pos, r=self.radius, lattice=lattice, loop=False
        )

        ys = row["targets"]
        y = torch.cat(ys) if ys else torch.empty(0)

        data = Data(x=types.unsqueeze(-1), pos=pos, edge_index=edge_index, y=y)
        data.lattice = lattice
        box, is_orthorhombic = _normalize_lattice(lattice)
        if is_orthorhombic:
            # Store as (1, 3) so PyG batching collates it to (B, 3); a bare
            # (3,) vector would collate to (B*3,), which breaks `box[batch]`
            # in the model's PBC minimum-image edge path.
            data.box = box.reshape(1, 3).clone()
        data.mol_name = str(idx)
        data.n_atoms = n_query_atoms
        if mol_number is not None:
            data.mol_number = mol_number
            data.mol_is_query = mol_is_query
            data.n_context_atoms = int(pos.shape[0]) - n_query_atoms
            data.n_context_molecules = int(len(n_per)) - 1

        # --- extra per-molecule data (attached when present in the file) ------
        # Vector quantities are stored as (1, 3) so PyG batching collates them
        # to (B, 3) -- the same convention as `box` above; a bare (3,) vector
        # would collate to (B*3,).
        if row["position"] is not None:
            data.com = row["position"].reshape(1, 3)
        else:
            data.com = pos[:n_query_atoms].mean(dim=0, keepdim=True)  # query centroid
        if row["orientation"] is not None:
            data.orientation = row["orientation"].reshape(1, 3)
        data.species = row["species"]
        if row["dipole_moment"] is not None:
            data.dipole_moment = row["dipole_moment"].reshape(1, 3)
        for field in ("S1_S0", "T1_S0"):
            val = row[f"transition_dipole_{field}"]
            if val is not None:
                data[f"transition_dipole_{field}"] = val.reshape(1, 3)
        if self._species_names:
            data.species_name = self._species_names[data.species]
        if self._species_smiles:
            data.species_smiles = self._species_smiles[data.species]

        logger.debug(
            "Box sample idx=%d: %d query atoms, %d context atoms, %d edges, com=%s",
            idx,
            data.n_atoms,
            getattr(data, "n_context_atoms", 0),
            data.edge_index.shape[1],
            data.com.tolist(),
        )
        return data


class CombinedBoxMolecularDataset(Dataset):
    """A PyTorch Geometric Dataset spanning several box HDF5 files.

    Combines one :class:`BoxMolecularDataset` per file into a single flat
    dataset, so samples behave exactly like the single-file case: each returns
    a ``Data`` with ``x`` (atom types), ``pos``, ``edge_index``, ``y``,
    ``com`` (center of mass), ``lattice``, ``box``, ``mol_name`` and the
    per-molecule properties (orientation, dipoles, species, ...).

    Args:
        h5_paths: A single path or a list of paths to box HDF5 files.
        target_key: Property key(s) per molecule -- a string, a list, or ``None``
            when only box-level access is needed (short name or full
            ``group/dataset`` path).
        radius: Radius for the periodic radius-graph construction.
        box_key: Key of the periodic box lattice within each file.
        keep_in_memory: When True, each per-file dataset keeps its HDF5 data in
            memory (see :class:`BoxMolecularDataset`).
        context: Optional per-molecule context configuration forwarded to every
            per-file dataset (see :class:`BoxMolecularDataset`); when set, each
            sample's graph also contains the surrounding molecules' atoms.
    """

    def __init__(
        self,
        h5_paths: str | list[str],
        target_key: str | list[str] | None = None,
        radius: float = 6.0,
        box_key: str = "lattice",
        keep_in_memory: bool = False,
        context: dict | None = None,
    ) -> None:
        if isinstance(h5_paths, str):
            h5_paths = [h5_paths]
        self.h5_paths = list(h5_paths)
        if not self.h5_paths:
            raise ValueError("h5_paths must contain at least one file")

        self.target_key = target_key
        self.radius = radius
        self.box_key = box_key
        self.keep_in_memory = bool(keep_in_memory)
        self.context = dict(context or {})

        # One sub-dataset per file; reuses all per-file loading logic so the
        # combined dataset behaves "in the same way" as the original one.
        self.datasets = [
            BoxMolecularDataset(
                path,
                target_key=target_key,
                radius=radius,
                box_key=box_key,
                keep_in_memory=keep_in_memory,
                context=context,
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
            "CombinedBoxMolecularDataset: %d file(s) -> %d sample(s); per-file=%s",
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

    def n_boxes(self) -> int:
        """Number of boxes (files) in the combined dataset."""
        return len(self.datasets)

    def box_sample(self, box_idx: int, radius: float | None = None) -> Data:
        """One box-level sample (molecules as nodes) from file ``box_idx``.

        Delegates to :meth:`BoxMolecularDataset.box_sample`, so the diffusion
        runner can iterate boxes without a dedicated diffusion dataset class.
        """
        return self.datasets[box_idx].box_sample(radius=radius)

    def box_reference(self, box_idx: int = 0) -> dict:
        """Per-molecule reference metadata for the box at file ``box_idx``."""
        return self.datasets[box_idx].box_reference()


class ZOrderedBoxMolecularDataset(Dataset):
    """Per-molecule diffusion samples for sequential (+z) film generation.

    Built on :class:`CombinedBoxMolecularDataset` (which in turn is built on
    :class:`BoxMolecularDataset`). Every molecule of every box becomes one
    sample: the *studied* molecule (the target) plus every molecule whose
    center-of-mass z is **at or below** its z — all molecules higher in z than
    the studied molecule are thrown away. The returned ``Data`` is a graph of
    the kept molecule COMs (nodes = molecules) with a per-node ``target_mask``
    marking the target z-block: the studied molecule plus the next-highest
    ``chunk_size - 1`` molecules below it (so ``chunk_size=1`` is exactly the
    studied molecule). The diffusion trainer never has to invent a z-frontier
    or expand the mask itself: it simply keeps the non-target nodes clean and
    denoises the target block (see ``DiffusionMoleculeModule._corrupt_z_ordered``).

    Because every molecule is a sample, the standard random train/val/test split
    (``runs.training_helpers.build_loaders``) draws random molecules across the
    whole film — the validation and test sets are not restricted to a z-slab of
    the box.

    For generation, :meth:`box_sample` / :meth:`box_reference` expose the full
    box (all molecules) so a whole new thin film can be generated/reconstructed.
    """

    def __init__(
        self,
        h5_paths: str | list[str],
        radius: float = 20.0,
        box_key: str = "lattice",
        keep_in_memory: bool = False,
        chunk_size: int = 1,
    ) -> None:
        if isinstance(h5_paths, str):
            h5_paths = [h5_paths]
        self.h5_paths = list(h5_paths)
        self.radius = radius
        self.box_key = box_key
        self.chunk_size = max(int(chunk_size), 1)

        self.molecular = CombinedBoxMolecularDataset(
            self.h5_paths,
            target_key=None,
            radius=radius,
            box_key=box_key,
            keep_in_memory=keep_in_memory,
        )
        # Precompute per-file COMs, species and lattice once (cheap accessors).
        self._coms = [ds.coms() for ds in self.molecular.datasets]
        self._species = [ds.species_ids() for ds in self.molecular.datasets]
        self._lattices = [ds.lattice for ds in self.molecular.datasets]
        self._mapping = [
            (fi, mi)
            for fi, ds in enumerate(self.molecular.datasets)
            for mi in range(len(ds))
        ]
        self._num_samples = len(self._mapping)
        super().__init__(root=None)
        logger.info(
            "ZOrderedBoxMolecularDataset: %d molecule(s) over %d file(s), "
            "radius=%.3f (per-molecule samples; context = molecules at/below "
            "the studied molecule's z)",
            self._num_samples,
            len(self.molecular.datasets),
            radius,
        )

    def __len__(self) -> int:
        return self._num_samples

    def mol_ids(self) -> list[str]:
        """Molecule identifier per sample (file-qualified molecule index)."""
        return [f"{fi}:{mi}" for fi, mi in self._mapping]

    def n_boxes(self) -> int:
        """Number of boxes (files) — used by the runner for generation."""
        return self.molecular.n_boxes()

    def box_sample(self, box_idx: int = 0, radius: float | None = None) -> Data:
        """Full-box sample (all molecules) used as generation reference/truth."""
        return self.molecular.box_sample(box_idx, radius=radius)

    def box_reference(self, box_idx: int = 0) -> dict:
        """Per-molecule reference metadata of the box (generation eval / viz)."""
        return self.molecular.box_reference(box_idx)

    def __getitem__(self, idx: int) -> Data:
        """One z-ordered sample: the studied molecule + everything at/below its z.

        Returns a ``Data`` of the kept molecule COMs with a PBC radius graph,
        ``box``/``lattice``, and a per-node ``target_mask`` marking the target
        z-block (the studied molecule plus the next-highest ``chunk_size - 1``
        molecules). Molecules higher in z than the studied molecule are thrown
        away — they are the ones the model must learn to generate later.
        """
        fi, mi = self._mapping[idx]
        coms = self._coms[fi]  # (N, 3)
        species = self._species[fi]  # (N,)
        lattice = self._lattices[fi]
        z = coms[:, 2]
        # Throw away every molecule higher in z than the studied molecule; keep
        # the studied molecule and everything at/below its z.
        keep = z <= z[mi]
        kept = torch.nonzero(keep).flatten()  # (k,)
        pos = coms[kept]
        x = species[kept].reshape(-1, 1)
        edge_index = try_cuda_radius_graph_pbc(
            pos, r=self.radius, lattice=lattice, loop=False
        )
        box, is_orthorhombic = _normalize_lattice(lattice)
        # The dataset owns the target-block mask: the studied molecule plus the
        # next-highest ``chunk_size - 1`` molecules below it (``chunk_size=1``
        # is exactly the studied molecule). The trainer just reads it.
        kept_z = coms[kept, 2]
        target_order = torch.argsort(kept_z, descending=True)
        target_mask = torch.zeros(len(kept), dtype=torch.bool)
        target_mask[target_order[: self.chunk_size]] = True
        data = Data(x=x, pos=pos, edge_index=edge_index, y=torch.zeros(1))
        data.box = box.reshape(1, 3).clone()
        data.lattice = lattice.clone()
        data.is_orthorhombic = torch.tensor([int(is_orthorhombic)], dtype=torch.long)
        data.target_mask = target_mask
        data.mol_name = self.mol_ids()[idx]
        return data


def get_z_ordered_box_dataset(
    h5_paths: str | list[str],
    radius: float = 20.0,
    box_key: str = "lattice",
    keep_in_memory: bool = False,
    chunk_size: int = 1,
) -> ZOrderedBoxMolecularDataset:
    """Helper function to instantiate a ZOrderedBoxMolecularDataset."""
    return ZOrderedBoxMolecularDataset(
        h5_paths=h5_paths,
        radius=radius,
        box_key=box_key,
        keep_in_memory=keep_in_memory,
        chunk_size=chunk_size,
    )


def get_box_dataset(
    h5_path: str,
    target_key: str | list[str],
    radius: float = 6.0,
    box_key: str = "lattice",
    keep_in_memory: bool = False,
    context: dict | None = None,
) -> BoxMolecularDataset:
    """Helper function to instantiate a BoxMolecularDataset."""
    return BoxMolecularDataset(
        h5_path=h5_path,
        target_key=target_key,
        radius=radius,
        box_key=box_key,
        keep_in_memory=keep_in_memory,
        context=context,
    )


def get_combined_box_dataset(
    h5_paths: str | list[str],
    target_key: str | list[str],
    radius: float = 6.0,
    box_key: str = "lattice",
    keep_in_memory: bool = False,
    context: dict | None = None,
) -> CombinedBoxMolecularDataset:
    """Helper function to instantiate a CombinedBoxMolecularDataset."""
    return CombinedBoxMolecularDataset(
        h5_paths=h5_paths,
        target_key=target_key,
        radius=radius,
        box_key=box_key,
        keep_in_memory=keep_in_memory,
        context=context,
    )
