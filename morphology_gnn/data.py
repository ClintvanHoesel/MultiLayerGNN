import torch
import os
import h5py
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import radius_graph
from .radius_graph import radius_graph_pbc

try:
    from .cuda_radius_graph import radius_graph_pbc as cuda_radius_graph_pbc

    _cuda_available = True
except Exception:
    cuda_radius_graph_pbc = None
    _cuda_available = False


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
            target_key (str): The key in the HDF5 group containing the target property (e.g., 'Positive VIP').
            radius (float): The radius for radius_graph construction.
            box_key (str): The key in the HDF5 group containing the periodic box lattice.
        """
        self.h5_path = h5_path
        self.target_key = target_key
        self.radius = radius
        self.box_key = box_key

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found at: {h5_path}")

        # Build a flat index of (group key, frame index) samples.
        self._index: list[tuple[str, int | None]] = []
        with h5py.File(self.h5_path, "r") as hf:
            for key in hf.keys():
                group = hf[key]
                if not isinstance(group, h5py.Group) or "pos" not in group:
                    continue
                if group["pos"].ndim == 3:
                    n_frames = group["pos"].shape[0]
                    self._index.extend((key, f) for f in range(n_frames))
                else:
                    self._index.append((key, None))

        self._num_samples = len(self._index)
        super().__init__(root=None)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Data:
        """
        Loads a single molecule frame's data and constructs the graph.
        """
        mol_key, frame = self._index[idx]

        with h5py.File(self.h5_path, "r") as hf:
            group = hf[mol_key]

            # Load positions and atom types for a single frame.
            if frame is not None:
                pos = torch.tensor(group["pos"][frame], dtype=torch.float)
                types = torch.tensor(group["types"][frame], dtype=torch.long)
            else:
                pos = torch.tensor(group["pos"][:], dtype=torch.float)
                types = torch.tensor(group["types"][:], dtype=torch.long)

            # Load target property.
            if frame is not None:
                y = torch.tensor(group[self.target_key][frame], dtype=torch.float)
            else:
                y = torch.tensor(group[self.target_key][:], dtype=torch.float)
            y = y.view(-1)

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

            data.mol_name = mol_key
            data.frame = frame if frame is not None else 0

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
        target_key: The property key within each group (e.g. 'Positive VIP').
        radius: Radius for radius_graph construction.
        box_key: Key of the periodic box lattice within each group.
    """

    def __init__(
        self,
        h5_paths: str | list[str],
        target_key: str,
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

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Data:
        """Load the sample at flat index ``idx``, delegating to the owning file."""
        dataset_idx, sample_idx = self._mapping[idx]
        return self.datasets[dataset_idx][sample_idx]

    def file_counts(self) -> list[int]:
        """Number of samples contributed by each file, in order."""
        return [len(ds) for ds in self.datasets]


def get_combined_h5_dataset(
    h5_paths: str | list[str],
    target_key: str,
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
