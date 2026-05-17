import torch
import os
import h5py
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import radius_graph

try:
    from .cuda_radius_graph import radius_graph_pbc as cuda_radius_graph_pbc
except Exception:
    cuda_radius_graph_pbc = None


def radius_graph_pbc(
    pos: torch.Tensor,
    r: float,
    lattice: torch.Tensor,
    loop: bool = False,
    max_num_neighbors: int | None = None,
) -> torch.Tensor:
    """Build a radius graph with periodic boundary conditions.

    Args:
        pos: Tensor of shape (N, 3) containing atomic positions.
        r: Radius cutoff.
        lattice: Either a 3-vector of box lengths or a 3x3 lattice matrix.
        loop: Whether to include self-loops.
        max_num_neighbors: Optional maximum number of neighbors for each node.
    """
    N = pos.size(0)
    device = pos.device
    lattice = lattice.to(device)

    if lattice.ndim == 1:
        if lattice.numel() != 3:
            raise ValueError("lattice must have shape (3,) or (3, 3)")
        box = lattice
        is_orthorhombic = True
    elif lattice.shape == (3, 3):
        box = torch.diagonal(lattice)
        is_orthorhombic = torch.allclose(lattice, torch.diag(box))
    else:
        raise ValueError("lattice must have shape (3,) or (3, 3)")

    if not is_orthorhombic:
        # Fall back to periodic-image construction for non-orthorhombic cells.
        shifts = torch.stack(
            torch.meshgrid(
                torch.tensor([-1, 0, 1], device=device),
                torch.tensor([-1, 0, 1], device=device),
                torch.tensor([-1, 0, 1], device=device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)

        pos_images = pos.unsqueeze(0) + shifts.unsqueeze(1) @ lattice
        pos_images = pos_images.reshape(-1, 3)

        edge_index = radius_graph(
            pos_images,
            r=r,
            loop=loop,
            max_num_neighbors=max_num_neighbors,
        )

        mask = edge_index[0] < N
        edge_index = edge_index[:, mask]
        edge_index[1] = edge_index[1] % N
        edge_index = torch.unique(edge_index, dim=1)
        return edge_index

    # Efficient minimum-image convention for orthorhombic boxes.
    box = box.to(device)
    pos = torch.remainder(pos, box)

    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    diff = diff - torch.round(diff / box) * box
    dist2 = (diff * diff).sum(dim=-1)

    if not loop:
        dist2.fill_diagonal_(float("inf"))

    if max_num_neighbors is None:
        mask = dist2 <= r * r
        i, j = torch.nonzero(mask, as_tuple=True)
        edge_index = torch.stack([i, j], dim=0)
        return edge_index

    # Keep the closest max_num_neighbors within the cutoff for each node.
    edge_list = []
    for src in range(N):
        valid = dist2[src] <= r * r
        if not valid.any():
            continue
        distances = dist2[src].clone()
        distances[~valid] = float("inf")
        if max_num_neighbors < valid.sum().item():
            idx = torch.topk(-distances, k=max_num_neighbors, largest=True).indices
            edge_list.append(
                torch.stack(
                    [torch.full((idx.size(0),), src, device=device), idx], dim=0
                )
            )
        else:
            idx = torch.nonzero(valid, as_tuple=True)[0]
            edge_list.append(
                torch.stack(
                    [torch.full((idx.size(0),), src, device=device), idx], dim=0
                )
            )

    if edge_list:
        edge_index = torch.cat(edge_list, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    return edge_index


class H5MolecularDataset(Dataset):
    """
    A PyTorch Geometric Dataset that loads molecular graphs directly from HDF5 files.
    This implementation reconstructs the graph structure (edge_index) on-the-fly
    making it extremely memory efficient.
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

        # Pre-count the number of molecule groups in the H5 file
        with h5py.File(self.h5_path, "r") as hf:
            self._molecule_keys = [
                key for key in hf.keys() if isinstance(hf[key], h5py.Group)
            ]
            self._num_samples = len(self._molecule_keys)

        super().__init__(root=None)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Data:
        """
        Loads a single molecule's data and constructs the graph.
        """
        mol_key = self._molecule_keys[idx]

        try:
            with h5py.File(self.h5_path, "r") as hf:
                group = hf[mol_key]

                # Load positions and atom types
                pos = torch.tensor(group["pos"][:], dtype=torch.float)
                types = torch.tensor(group["types"][:], dtype=torch.long)

                # Load target property
                y = torch.tensor(group[self.target_key][:], dtype=torch.float).view(-1)

                # Load box lattice if available and build a periodic radius graph.
                lattice = None
                if self.box_key in group:
                    lattice = torch.tensor(group[self.box_key][:], dtype=torch.float)

                if lattice is not None:
                    if cuda_radius_graph_pbc is not None:
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

                # Create the PyG Data object
                data = Data(x=types.unsqueeze(-1), pos=pos, edge_index=edge_index, y=y)

                if lattice is not None:
                    data.lattice = lattice

                data.mol_name = mol_key

                return data

        except Exception as e:
            print(f"Error loading molecule {mol_key} at index {idx}: {e}")
            return Data()


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
