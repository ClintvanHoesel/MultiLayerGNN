import logging
import os

import h5py
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import radius_graph

from .radius_graph import radius_graph_pbc

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
