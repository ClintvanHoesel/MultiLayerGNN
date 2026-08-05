import torch


def _select_edges_from_dist2(
    dist2: torch.Tensor,
    r: float,
    loop: bool,
    max_num_neighbors: int | None,
) -> torch.Tensor:
    """Turn an (N, N) squared-distance matrix into a directed edge_index.

    Uses the same neighbor-selection semantics everywhere so the
    orthorhombic and non-orthorhombic paths behave identically:

    * ``loop=False`` excludes self-edges.
    * ``max_num_neighbors=None`` keeps every pair within the cutoff.
    * Otherwise, the closest ``max_num_neighbors`` neighbors within the
      cutoff are kept for each node.
    """
    device = dist2.device
    if not loop:
        dist2 = dist2.clone()
        dist2.fill_diagonal_(float("inf"))

    if max_num_neighbors is None:
        mask = dist2 <= r * r
        i, j = torch.nonzero(mask, as_tuple=True)
        return torch.stack([i, j], dim=0)

    # Keep the closest max_num_neighbors within the cutoff for each node.
    edge_list = []
    for src in range(dist2.size(0)):
        valid = dist2[src] <= r * r
        if not valid.any():
            continue
        distances = dist2[src].clone()
        distances[~valid] = float("inf")
        if max_num_neighbors < valid.sum().item():
            idx = torch.topk(-distances, k=max_num_neighbors, largest=True).indices
        else:
            idx = torch.nonzero(valid, as_tuple=True)[0]
        edge_list.append(
            torch.stack([torch.full((idx.size(0),), src, device=device), idx], dim=0)
        )

    if edge_list:
        return torch.cat(edge_list, dim=1)
    return torch.empty((2, 0), dtype=torch.long, device=device)


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
        # General-lattice minimum-image via the 27 periodic images of the cell.
        # Wrap positions into the unit cell so the +/-1 shifts span the whole
        # cell (requires a cell larger than 2 * r for the minimum image to be
        # unambiguous, the same assumption as the orthorhombic path).
        lattice = lattice.to(pos.dtype)
        inv_lattice = torch.linalg.inv(lattice)
        frac = pos @ inv_lattice
        pos = (frac - torch.floor(frac)) @ lattice

        shifts = (
            torch.stack(
                torch.meshgrid(
                    torch.tensor([-1, 0, 1], device=device),
                    torch.tensor([-1, 0, 1], device=device),
                    torch.tensor([-1, 0, 1], device=device),
                    indexing="ij",
                ),
                dim=-1,
            )
            .reshape(-1, 3)
            .to(lattice.dtype)
        )

        shift_disps = shifts @ lattice  # (27, 3)

        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 3)
        dist2 = torch.full((N, N), float("inf"), dtype=pos.dtype, device=device)
        for s in range(shift_disps.size(0)):
            d2 = ((diff - shift_disps[s]) ** 2).sum(dim=-1)
            dist2 = torch.minimum(dist2, d2)

        return _select_edges_from_dist2(dist2, r, loop, max_num_neighbors)

    # Efficient minimum-image convention for orthorhombic boxes.
    box = box.to(device)
    pos = torch.remainder(pos, box)

    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    diff = diff - torch.round(diff / box) * box
    dist2 = (diff * diff).sum(dim=-1)

    return _select_edges_from_dist2(dist2, r, loop, max_num_neighbors)
