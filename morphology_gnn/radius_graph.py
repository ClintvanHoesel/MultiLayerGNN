import logging

import torch

logger = logging.getLogger(__name__)


def _normalize_lattice(lattice: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Resolve a lattice spec into ``(box, is_orthorhombic)``.

    Accepts a 3-vector of box lengths (orthorhombic) or a ``(3, 3)`` lattice
    matrix. For a ``(3, 3)`` matrix that is (numerically) diagonal the box
    lengths are returned; otherwise ``box`` is the diagonal (used for wrapping)
    and ``is_orthorhombic`` is ``False`` so callers can fall back to the general
    fractional-coordinate minimum image.
    """
    if lattice.ndim == 1:
        if lattice.numel() != 3:
            raise ValueError("lattice must have shape (3,) or (3, 3)")
        box = lattice
        is_orthorhombic = True
    elif lattice.ndim == 2 and lattice.shape == (3, 3):
        box = torch.diagonal(lattice)
        is_orthorhombic = torch.allclose(lattice, torch.diag(box))
    else:
        raise ValueError("lattice must have shape (3,) or (3, 3)")
    return box, is_orthorhombic


def wrap_pos(pos: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Wrap Cartesian positions into the periodic cell (orthorhombic).

    Args:
        pos: Node positions, shape ``(..., 3)``.
        box: Box lengths, shape ``(3,)`` (or broadcastable, e.g. per-node).

    Returns:
        Wrapped positions, same shape as ``pos``, each coordinate in ``[0, box)``.
    """
    return torch.remainder(pos, box.to(pos.dtype))


def min_image_disp(
    pos: torch.Tensor, edge_index: torch.Tensor, lattice: torch.Tensor
) -> torch.Tensor:
    """Minimum-image displacement vectors for an edge list under PBC.

    For each edge ``(src, dst)`` returns the shortest periodic displacement
    ``pos[dst] - pos[src]``, computed in fractional coordinates (``round`` the
    fractional difference so the image lies in the same cell). Correct for both
    orthorhombic boxes (3-vector or diagonal matrix) and general ``(3, 3)``
    lattice matrices. Requires the cell to be large enough for the minimum image
    to be unambiguous (cell > 2 * cutoff, the same assumption as
    :func:`radius_graph_pbc`).

    Args:
        pos: Node positions, shape ``(N, 3)``.
        edge_index: Connectivity, shape ``(2, E)``.
        lattice: ``(3,)`` box lengths or ``(3, 3)`` lattice matrix.

    Returns:
        Displacement vectors of shape ``(E, 3)``.
    """
    box, is_orthorhombic = _normalize_lattice(lattice)
    device = pos.device
    box = box.to(pos.dtype).to(device)
    src, dst = edge_index[0], edge_index[1]
    disp = pos[dst] - pos[src]  # (E, 3)
    if is_orthorhombic:
        return disp - torch.round(disp / box) * box
    # General lattice: minimum image in fractional coordinates.
    inv_lattice = torch.linalg.inv(lattice.to(pos.dtype).to(device))
    frac = disp @ inv_lattice  # (E, 3)
    frac = frac - torch.round(frac)
    return frac @ lattice.to(pos.dtype).to(device)


def rebuild_pbc_edges(
    batch_pos: torch.Tensor,
    batch_vec: torch.Tensor,
    cell: torch.Tensor,
    radius: float,
    loop: bool = False,
    max_num_neighbors: int | None = None,
) -> torch.Tensor:
    """Rebuild a batched PBC radius graph from (possibly noisy) positions.

    The graph depends on the atomic positions, so denoising steps must rebuild
    the edge list from the current (noisy) coordinates each time — this helper
    does that for a PyG-style batched sample, one graph at a time (molecules are
    small, so the per-graph loop is cheap).

    Args:
        batch_pos: Concatenated node positions, shape ``(N_total, 3)``.
        batch_vec: Node -> graph assignment, shape ``(N_total,)`` (PyG ``batch``).
        cell: Per-graph cell — ``(B, 3)`` box lengths or ``(B, 3, 3)`` lattice
            matrices.
        radius: Radius cutoff.
        loop: Whether to include self-loops.
        max_num_neighbors: Optional per-node neighbor cap.

    Returns:
        ``edge_index`` of shape ``(2, E_total)`` in the global (batched) node
        numbering.
    """
    device = batch_pos.device
    cell = cell.to(device)
    if cell.ndim == 2 and cell.shape[1] == 3:
        boxes, cell_mats = cell, None
    elif cell.ndim == 3 and cell.shape[1:] == (3, 3):
        boxes = torch.diagonal(cell, dim1=-2, dim2=-1)
        cell_mats = cell
    else:
        raise ValueError("cell must have shape (B, 3) or (B, 3, 3)")

    edge_list = []
    node_offset = 0
    B = boxes.shape[0]
    for i in range(B):
        mask = batch_vec == i
        n = int(mask.sum().item())
        if n == 0:
            continue
        sub_pos = batch_pos[mask]
        lattice = cell_mats[i] if cell_mats is not None else boxes[i]
        sub_edge = radius_graph_pbc(
            sub_pos,
            r=radius,
            lattice=lattice,
            loop=loop,
            max_num_neighbors=max_num_neighbors,
        )
        edge_list.append(sub_edge + node_offset)
        node_offset += n
    if edge_list:
        return torch.cat(edge_list, dim=1)
    return torch.empty((2, 0), dtype=torch.long, device=device)


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

    logger.debug(
        "radius_graph_pbc n=%d r=%.3f path=%s",
        N,
        r,
        "orthorhombic" if is_orthorhombic else "27-image",
    )

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

        edge_index = _select_edges_from_dist2(dist2, r, loop, max_num_neighbors)
        logger.debug("radius_graph_pbc: %d edges (27-image path)", edge_index.shape[1])
        return edge_index

    # Efficient minimum-image convention for orthorhombic boxes.
    box = box.to(device)
    pos = torch.remainder(pos, box)

    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    diff = diff - torch.round(diff / box) * box
    dist2 = (diff * diff).sum(dim=-1)

    edge_index = _select_edges_from_dist2(dist2, r, loop, max_num_neighbors)
    logger.debug("radius_graph_pbc: %d edges (orthorhombic path)", edge_index.shape[1])
    return edge_index
