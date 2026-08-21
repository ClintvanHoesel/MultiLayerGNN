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
    # General lattice: minimum image in fractional coordinates. DOES NOT ALWAYS WORK
    inv_lattice = torch.linalg.inv(lattice.to(pos.dtype).to(device))
    frac = disp @ inv_lattice  # (E, 3)
    frac = frac - torch.round(frac)
    return frac @ lattice.to(pos.dtype).to(device)


def wrap_pos_general(pos: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """Wrap Cartesian positions into the periodic cell for a general ``(3, 3)`` lattice.

    Positions are mapped to fractional coordinates, wrapped into ``[0, 1)`` and
    mapped back to Cartesian — the generalization of :func:`wrap_pos` to
    non-orthorhombic cells.
    """
    lattice = lattice.to(pos.dtype).to(pos.device)
    inv_lattice = torch.linalg.inv(lattice)
    frac = pos @ inv_lattice
    frac = frac - torch.floor(frac)
    return frac @ lattice


def _min_image_delta(
    delta: torch.Tensor,
    lattice: torch.Tensor,
    box: torch.Tensor,
    is_orthorhombic: bool,
) -> torch.Tensor:
    """Shortest periodic displacement for a difference vector ``delta``.

    Same minimum-image convention as :func:`min_image_disp`: orthorhombic boxes
    use the vectorized ``round`` trick, general lattices the fractional-coordinate
    rounding. The caller decides which path via ``is_orthorhombic`` (from
    :func:`_normalize_lattice`) so the two are consistent with the rest of the
    PBC machinery.
    """
    if is_orthorhombic:
        box = box.to(delta.dtype).to(delta.device)
        return delta - torch.round(delta / box) * box
    lattice = lattice.to(delta.dtype).to(delta.device)
    inv_lattice = torch.linalg.inv(lattice)
    frac = delta @ inv_lattice
    frac = frac - torch.round(frac)
    return frac @ lattice


def _unwrap_by_reference(
    pos: torch.Tensor, lattice: torch.Tensor, box: torch.Tensor, is_orthorhombic: bool
) -> torch.Tensor:
    """Minimum-image unwrap of every atom relative to the wrapped geometric centroid.

    The wrapped centroid is a point inside the cell around which the molecule is
    roughly centered; bringing each atom to the image nearest it "pulls the
    molecule together" into a contiguous blob. Requires the cell to be larger
    than the molecule's extent (else the minimum image is ambiguous).
    """
    c = pos.mean(dim=0)
    delta = _min_image_delta(pos - c, lattice, box, is_orthorhombic)
    return c + delta


def _unwrap_by_bonds(
    pos: torch.Tensor,
    bonds: torch.Tensor,
    lattice: torch.Tensor,
    box: torch.Tensor,
    is_orthorhombic: bool,
) -> torch.Tensor:
    """Unwrap by walking the bond graph: place each atom next to a placed bonded neighbor.

    Starts a new connected component at every not-yet-visited atom, so partial or
    multi-component bond lists still produce a valid unwrap (components are kept
    at their wrapped positions relative to one another). This is more robust than
    :func:`_unwrap_by_reference` when the molecule spans more than half the cell.
    """
    N = pos.shape[0]
    arr = bonds.to(dtype=torch.long, device=pos.device) if bonds.dim() == 2 else bonds
    adj: list[list[int]] = [[] for _ in range(N)]
    for a, b in zip(arr[:, 0].tolist(), arr[:, 1].tolist()):
        if a != b and 0 <= a < N and 0 <= b < N:
            adj[a].append(b)
            adj[b].append(a)
    out = pos.clone()
    visited = torch.zeros(N, dtype=torch.bool, device=pos.device)
    for start in range(N):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if visited[v]:
                    continue
                delta = _min_image_delta(pos[v] - out[u], lattice, box, is_orthorhombic)
                out[v] = out[u] + delta
                visited[v] = True
                stack.append(v)
    return out


def unwrap_molecule(
    pos: torch.Tensor,
    lattice: torch.Tensor,
    bonds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unwrap a PBC-wrapped molecule into a contiguous spatial arrangement.

    Atoms that sit across the periodic boundary are pulled back next to the rest
    of the molecule so the whole molecule forms a connected blob. When a bond
    list is provided the connectivity graph is walked (:func:`_unwrap_by_bonds`)
    to place every atom next to an already-placed bonded neighbor — the most
    robust path when a molecule spans more than half the cell. Otherwise each
    atom is brought to the minimum-image position relative to the wrapped
    geometric centroid (:func:`_unwrap_by_reference`).

    Args:
        pos: Wrapped atom positions, shape ``(N, 3)``.
        lattice: ``(3,)`` box lengths or ``(3, 3)`` lattice matrix.
        bonds: Optional connectivity — ``(E, 2)`` index pairs. Used for the
            graph-based unwrap when provided.

    Returns:
        Unwrapped positions of shape ``(N, 3)`` (the overall translation is
        arbitrary; only the shape is meaningful).
    """
    pos = pos.float()
    lattice = lattice.to(dtype=pos.dtype, device=pos.device)
    box, is_orthorhombic = _normalize_lattice(lattice)
    box = box.to(dtype=pos.dtype, device=pos.device)
    if bonds is not None:
        arr = (
            bonds
            if isinstance(bonds, torch.Tensor)
            else torch.as_tensor(bonds, dtype=torch.long, device=pos.device)
        )
        if arr.numel() > 0:
            return _unwrap_by_bonds(pos, arr, lattice, box, is_orthorhombic)
    return _unwrap_by_reference(pos, lattice, box, is_orthorhombic)


def pbc_center_of_mass(
    pos: torch.Tensor,
    lattice: torch.Tensor,
    masses: torch.Tensor | None = None,
    bonds: torch.Tensor | None = None,
    wrap: bool = True,
) -> torch.Tensor:
    """Mass-weighted center of mass of a molecule under periodic boundary conditions.

    The molecule is first unwrapped (:func:`unwrap_molecule`) so atoms that
    cross the periodic boundary are brought back to a contiguous spatial
    arrangement, then the (mass-weighted) mean position is computed. When
    ``wrap=True`` the result is folded back into the cell so it can be compared
    with the values stored under ``molecules/position`` in the SCM-pure HDF5
    files.

    Args:
        pos: Wrapped atom positions, shape ``(N, 3)``.
        lattice: ``(3,)`` box lengths or ``(3, 3)`` lattice matrix.
        masses: Per-atom weights, shape ``(N,)``. ``None`` computes the plain
            geometric centroid.
        bonds: Optional connectivity ``(E, 2)`` index pairs for the graph-based
            unwrap.
        wrap: Fold the COM back into the cell ``[0, box)``.

    Returns:
        The center of mass, shape ``(3,)``.
    """
    unwrapped = unwrap_molecule(pos, lattice, bonds=bonds)
    if masses is None:
        com = unwrapped.mean(dim=0)
    else:
        m = torch.as_tensor(
            masses, dtype=unwrapped.dtype, device=unwrapped.device
        ).reshape(-1)
        if m.numel() != unwrapped.shape[0]:
            raise ValueError(
                f"masses must have one entry per atom ({unwrapped.shape[0]}); "
                f"got {m.numel()}"
            )
        com = (unwrapped * m.unsqueeze(-1)).sum(dim=0) / m.sum().clamp_min(1e-12)
    if wrap:
        box, is_orthorhombic = _normalize_lattice(lattice)
        box = box.to(dtype=com.dtype, device=com.device)
        com = wrap_pos(com, box) if is_orthorhombic else wrap_pos_general(com, lattice)
    return com


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
