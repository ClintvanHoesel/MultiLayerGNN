"""Hierarchical molecular GNN: alternating atomistic and centre-of-mass (COM)
message passing.

This module adds a two-level hierarchy on top of the existing atomistic GNN
conventions (see :class:`~morphology_gnn.model.scaler_model.ScalarMoleculeModel`)
*without* modifying that model:

1. **Atomistic level** — atoms exchange messages over the existing atomistic
   graph (``edge_index``, with optional geometric ``edge_attr``), using the same
   conv stack / :class:`Residual` conventions as ``ScalarMoleculeModel``.
2. **Molecular / COM level** — every molecule is pooled into a COM node
   (:class:`AtomToCOM`), COM nodes exchange messages over a radius graph built
   from the molecular COM positions (:func:`build_com_graph` /
   :class:`COMMessagePassing`), and the updated molecular representation is fed
   back to the atoms of that molecule (:class:`COMToAtom`).

Each :class:`HierarchicalBlock` performs one full cycle::

    atoms --Atomistic GNN--> atoms --AtomToCOM--> COM
      ^                                              |
      |                                              v
      +-------- COMToAtom <----- COMMessagePassing --+

and the whole cycle is repeated ``num_hierarchical_layers`` times with
*independent* blocks per cycle (each block has its own weights).

Molecular identity is taken from the per-node ``mol_number`` tensor already
produced by the box context-mode dataset (``BoxMolecularDataset(context=...)``).
After PyG batching ``mol_number`` is treated as an *opaque* grouping key (PyG
offsets any ``*index`` node attribute by per-sample node counts), so it is
re-indexed into contiguous, batch-unique ids inside ``forward``.

Geometric conventions match the existing scalar GNN, so the model stays
SE(3)-invariant: node features are scalars, edge features are RBF-embedded
(minimum-image) distances via :class:`EdgeVectorLayer`, and COM positions are
used only to build the COM graph and its distance features — never as absolute
scalar features. COM positions are computed in-model, mass-weighted and
PBC-unwrapped, by reusing :func:`morphology_gnn.radius_graph.pbc_center_of_mass`
and the :class:`~morphology_gnn.periodic_table.PT` atomic weights.
"""

from __future__ import annotations

import inspect
import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, radius_graph
from torch_geometric.nn.aggr import MeanAggregation, MultiAggregation
from torch_geometric.utils import scatter

from .aggregator import build_aggregators
from .embedding import AtomTypeEmbedding, EdgeVectorLayer
from .residual import NORM_REGISTRY, Residual
from morphology_gnn.periodic_table import PT
from morphology_gnn.radius_graph import pbc_center_of_mass, rebuild_pbc_edges

logger = logging.getLogger(__name__)

# Largest atomic number in the standard periodic table (Oganesson, Z=118);
# matches AtomTypeEmbedding.MAX_ATOMIC_NUMBER.
_MAX_ATOMIC_NUMBER = 118


class AtomToCOM(nn.Module):
    """Aggregate atom representations into per-molecule COM node representations.

    Args:
        hidden_dim: Atom feature width.
        com_hidden_channels: COM feature width (projection output).
        aggregation: ``"mean"`` (default), ``"sum"`` or ``"attention"`` — the
            permutation-invariant pooling applied to the atoms of each molecule.

    The aggregation is a scatter over the batch-unique molecule key, so atoms of
    different molecules never mix. ``"attention"`` uses a learned per-atom score
    (softmax-normalised within each molecule) before the weighted sum. The
    pooled vector is projected to ``com_hidden_channels``.
    """

    def __init__(
        self,
        hidden_dim: int,
        com_hidden_channels: int,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        if aggregation not in ("mean", "sum", "attention"):
            raise ValueError(
                f"com_aggregation must be one of mean/sum/attention, got "
                f"{aggregation!r}"
            )
        self.aggregation = aggregation
        self.proj: nn.Linear = nn.Linear(hidden_dim, com_hidden_channels)
        if aggregation == "attention":
            self.att_mlp: nn.Sequential | None = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.att_mlp = None

    def reset_parameters(self) -> None:
        self.proj.reset_parameters()
        if self.att_mlp is not None:
            for m in self.att_mlp:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

    def forward(
        self,
        h_atoms: torch.Tensor,
        mol_key: torch.Tensor,
        dim_size: int,
    ) -> torch.Tensor:
        """Pool atoms -> COM features.

        Args:
            h_atoms: Atom features, shape ``(N, hidden_dim)``.
            mol_key: Batch-unique molecule id per atom, shape ``(N,)``.
            dim_size: Number of COM nodes ``M``.

        Returns:
            COM features of shape ``(M, com_hidden_channels)``.
        """
        if self.aggregation == "attention":
            att_mlp = self.att_mlp
            assert att_mlp is not None
            logits = att_mlp(h_atoms).squeeze(-1)  # (N,)
            denom = scatter(
                logits.exp(), mol_key, dim=0, dim_size=dim_size, reduce="sum"
            )  # (M,)
            weights = logits.exp() / denom[mol_key].clamp_min(1e-12)  # (N,)
            h_com = scatter(
                h_atoms * weights.unsqueeze(-1),
                mol_key,
                dim=0,
                dim_size=dim_size,
                reduce="sum",
            )
        else:
            h_com = scatter(
                h_atoms, mol_key, dim=0, dim_size=dim_size, reduce=self.aggregation
            )
        return self.proj(h_com)


def filter_intra_molecular_edges(
    edge_index: torch.Tensor,
    mol_number: torch.Tensor,
) -> torch.Tensor:
    """Keep only atomistic edges whose endpoints belong to the same molecule.

    Inter-molecular (cross-molecule) edges are dropped so that molecules do NOT
    exchange information at the atomistic level; cross-molecule communication is
    handled exclusively by the COM–COM graph (:func:`build_com_graph`). Both
    tensors use the same (possibly PyG-collated) node numbering.

    Args:
        edge_index: Atomistic connectivity, shape ``(2, E)``.
        mol_number: Molecule id per node, shape ``(N,)``.

    Returns:
        The intra-molecular subset of ``edge_index``, shape ``(2, E_intra)``.
    """
    src, dst = edge_index
    keep = mol_number[src] == mol_number[dst]
    return edge_index[:, keep]


def build_com_graph(
    com_pos: torch.Tensor,
    com_batch: torch.Tensor,
    box: torch.Tensor | None,
    com_cutoff: float,
    loop: bool = False,
    max_num_neighbors: int | None = None,
    mode: str = "radius",
) -> torch.Tensor:
    """Build the molecular COM-level graph (batched, per-graph separated).

    ``mode="radius"`` connects every pair of COM nodes within ``com_cutoff``,
    reusing :func:`morphology_gnn.radius_graph.rebuild_pbc_edges` for the
    periodic case (one radius graph per batch sample, with node offsets) and
    :func:`torch_geometric.nn.radius_graph` for non-periodic samples.
    ``mode="all"`` builds a *fully-connected* COM graph within each batch
    sample, so every molecule in the sample (i.e. all molecules in a large
    vicinity) participates in COM–COM interactions regardless of distance. In
    both modes molecules of *different* graphs never connect.

    Args:
        com_pos: COM node positions, shape ``(M, 3)`` (concatenated over graphs).
        com_batch: COM node -> graph assignment, shape ``(M,)`` (values ``0..B-1``).
        box: Per-graph cell — ``(B, 3)`` box lengths or ``(B, 3, 3)`` lattice
            matrices; ``None`` for non-periodic graphs.
        com_cutoff: COM–COM radius cutoff (``mode="radius"`` only).
        loop: Whether to include self-loops.
        max_num_neighbors: Optional per-node neighbor cap (``mode="radius"``).
        mode: ``"radius"`` (default) or ``"all"``.

    Returns:
        ``com_edge_index`` of shape ``(2, E)`` in the global (batched) COM node
        numbering.
    """
    device = com_pos.device
    if mode == "all":
        edge_list = []
        node_offset = 0
        for g in range(int(com_batch.max().item()) + 1):
            mask = com_batch == g
            n = int(mask.sum().item())
            if n > 1:
                idx = torch.arange(n, device=device)
                pairs = torch.cartesian_prod(idx, idx).t()  # (2, n*n)
                if not loop:
                    pairs = pairs[:, pairs[0] != pairs[1]]
                edge_list.append(pairs + node_offset)
            node_offset += n
        if edge_list:
            return torch.cat(edge_list, dim=1)
        return torch.empty((2, 0), dtype=torch.long, device=device)
    if mode != "radius":
        raise ValueError(f"com_graph mode must be 'radius' or 'all', got {mode!r}")
    if box is not None:
        return rebuild_pbc_edges(
            com_pos,
            com_batch,
            box,
            radius=com_cutoff,
            loop=loop,
            max_num_neighbors=max_num_neighbors,
        )
    else:
        raise Exception("Box needs to be available.")
    # Non-periodic fallback: per-graph radius graph with node offsets.
    edge_list = []
    node_offset = 0
    for g in range(int(com_batch.max().item()) + 1):
        mask = com_batch == g
        n = int(mask.sum().item())
        if n == 0:
            continue
        # PyG's radius_graph requires an int cap; keep every pair within the
        # cutoff (consistent with the PBC path) unless a cap was requested.
        max_nn = max_num_neighbors if max_num_neighbors is not None else max(n - 1, 1)
        sub_edge = radius_graph(
            com_pos[mask],
            r=com_cutoff,
            loop=loop,
            max_num_neighbors=max_nn,
        )
        edge_list.append(sub_edge + node_offset)
        node_offset += n
    if edge_list:
        return torch.cat(edge_list, dim=1)
    return torch.empty((2, 0), dtype=torch.long, device=device)


class COMMessagePassing(nn.Module):
    """Message-passing between COM nodes (molecules).

    Mirrors the atomistic conv stack of :class:`ScalarMoleculeModel`: a stack of
    ``com_num_layers`` convs of the configured ``conv_class`` (default
    ``GATConv``), each wrapped in a :class:`Residual` connection, operating on
    the COM graph with optional geometric edge attributes.
    """

    def __init__(
        self,
        com_hidden_channels: int,
        com_num_layers: int = 1,
        conv_class=GATConv,
        conv_kwargs: dict | None = None,
        act=F.gelu,
        dropout: float = 0.0,
        use_residual: bool = True,
        residual_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if com_num_layers < 1:
            raise ValueError(f"com_num_layers must be >= 1, got {com_num_layers}")
        self.com_hidden_channels = com_hidden_channels
        self.com_num_layers = com_num_layers
        self.act = act
        self.dropout = dropout
        conv_kwargs = dict(conv_kwargs or {})
        if conv_class is GATConv:
            conv_kwargs.setdefault("heads", 3)
            conv_kwargs.setdefault("concat", False)
        self.use_residual = use_residual
        residual_kwargs = dict(residual_kwargs or {})
        if use_residual:
            self.convs: nn.ModuleList = nn.ModuleList(
                [
                    Residual(
                        conv_class(
                            com_hidden_channels, com_hidden_channels, **conv_kwargs
                        ),
                        hidden_dim=com_hidden_channels,
                        **residual_kwargs,
                    )
                    for _ in range(com_num_layers)
                ]
            )
        else:
            self.convs = nn.ModuleList(
                [
                    conv_class(com_hidden_channels, com_hidden_channels, **conv_kwargs)
                    for _ in range(com_num_layers)
                ]
            )

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        h_com: torch.Tensor,
        com_edge_index: torch.Tensor,
        com_edge_attr: torch.Tensor | None = None,
        com_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the COM-level message passing.

        Args:
            h_com: COM features, shape ``(M, com_hidden_channels)``.
            com_edge_index: COM connectivity, shape ``(2, E)``.
            com_edge_attr: Optional geometric edge attributes, shape ``(E, num_rbf)``.
            com_batch: COM node -> graph assignment (for per-graph norms).

        Returns:
            Updated COM features, shape ``(M, com_hidden_channels)``.
        """
        for i, conv in enumerate(self.convs):
            if self.use_residual:
                # The residual wrapper consumes `batch` for per-graph
                # normalization (it never reaches the conv sublayer).
                if com_edge_attr is not None:
                    h_com = conv(
                        h_com, com_edge_index, edge_attr=com_edge_attr, batch=com_batch
                    )
                else:
                    h_com = conv(h_com, com_edge_index, batch=com_batch)
            else:
                if com_edge_attr is not None:
                    h_com = conv(h_com, com_edge_index, edge_attr=com_edge_attr)
                else:
                    h_com = conv(h_com, com_edge_index)
            h_com = self.act(h_com)
            if i < self.com_num_layers - 1:
                h_com = F.dropout(h_com, p=self.dropout, training=self.training)
        return h_com


class COMToAtom(nn.Module):
    """Project the updated molecular (COM) representation back onto its atoms.

    For every atom ``i`` of molecule ``m`` (``atom_mol[i] == m``) this computes a
    learnable residual update from the COM representation ``h_com[m]`` and adds
    it to the atom feature — preserving the existing atomistic representation.
    With ``gated=True`` (default) the update is modulated by a learned gate
    conditioned on both the atom and COM features::

        h_i <- h_i + sigmoid(gate(h_i || h_com[m])) * proj(h_com[m])

    ``gated=False`` uses the plain residual ``h_i <- h_i + proj(h_com[m])``.
    """

    def __init__(
        self,
        com_hidden_channels: int,
        hidden_dim: int,
        gated: bool = True,
    ) -> None:
        super().__init__()
        self.gated = bool(gated)
        self.proj = nn.Linear(com_hidden_channels, hidden_dim)
        if self.gated:
            self.gate = nn.Linear(hidden_dim + com_hidden_channels, hidden_dim)

    def reset_parameters(self) -> None:
        self.proj.reset_parameters()
        if self.gated:
            self.gate.reset_parameters()

    def forward(
        self,
        h_atoms: torch.Tensor,
        h_com: torch.Tensor,
        atom_mol: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast the COM representation back to its atoms (residual).

        Args:
            h_atoms: Atom features, shape ``(N, hidden_dim)``.
            h_com: COM features, shape ``(M, com_hidden_channels)``.
            atom_mol: Molecule id per atom, shape ``(N,)`` (``0..M-1``).

        Returns:
            Updated atom features, shape ``(N, hidden_dim)``.
        """
        com_broadcast = h_com[atom_mol]  # (N, com_hidden_channels)
        update = self.proj(com_broadcast)  # (N, hidden_dim)
        if self.gated:
            gate = torch.sigmoid(self.gate(torch.cat([h_atoms, com_broadcast], dim=-1)))
            update = gate * update
        return h_atoms + update


class HierarchicalBlock(nn.Module):
    """One full hierarchical cycle: atoms -> COM -> COM MP -> atoms.

    Runs the atomistic message-passing stack first (``num_layers`` convs over
    the atomistic graph), pools the atoms of each molecule into a COM node
    (:class:`AtomToCOM`), runs ``com_num_layers`` COM message-passing layers,
    then feeds the updated molecular representation back to the atoms
    (:class:`COMToAtom`).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        com_hidden_channels: int,
        com_num_layers: int,
        conv_class,
        conv_kwargs: dict | None,
        act,
        dropout: float,
        com_aggregation: str,
        use_residual: bool,
        residual_kwargs: dict | None,
        com_residual_kwargs: dict | None,
        gated: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.act = act
        self.dropout = dropout
        self.use_residual = use_residual
        residual_kwargs = dict(residual_kwargs or {})
        com_residual_kwargs = dict(com_residual_kwargs or {})
        conv_kwargs = dict(conv_kwargs or {})
        # Atomistic conv stack (mirrors ScalarMoleculeModel).
        if use_residual:
            self.atom_convs: nn.ModuleList = nn.ModuleList(
                [
                    Residual(
                        conv_class(hidden_dim, hidden_dim, **conv_kwargs),
                        hidden_dim=hidden_dim,
                        **residual_kwargs,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.atom_convs = nn.ModuleList(
                [
                    conv_class(hidden_dim, hidden_dim, **conv_kwargs)
                    for _ in range(num_layers)
                ]
            )
        self.atom_to_com = AtomToCOM(
            hidden_dim, com_hidden_channels, aggregation=com_aggregation
        )
        self.com_mp = COMMessagePassing(
            com_hidden_channels=com_hidden_channels,
            com_num_layers=com_num_layers,
            conv_class=conv_class,
            conv_kwargs=conv_kwargs,
            act=act,
            dropout=dropout,
            use_residual=use_residual,
            residual_kwargs=com_residual_kwargs,
        )
        self.com_to_atom = COMToAtom(com_hidden_channels, hidden_dim, gated=gated)

    def reset_parameters(self) -> None:
        for conv in self.atom_convs:
            conv.reset_parameters()
        self.atom_to_com.reset_parameters()
        self.com_mp.reset_parameters()
        self.com_to_atom.reset_parameters()

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        batch: torch.Tensor,
        atom_mol: torch.Tensor,
        dim_size: int,
        com_edge_index: torch.Tensor,
        com_edge_attr: torch.Tensor | None,
        com_batch: torch.Tensor,
    ) -> torch.Tensor:
        """One hierarchical cycle.

        Args:
            h: Atom features ``(N, hidden_dim)``.
            edge_index: Atomistic connectivity ``(2, E)``.
            edge_attr: Optional atomistic edge attributes ``(E, num_rbf)``.
            batch: Atom -> graph assignment ``(N,)``.
            atom_mol: Molecule id per atom ``(N,)`` (``0..M-1``).
            dim_size: Number of COM nodes ``M``.
            com_edge_index: COM connectivity ``(2, E_c)``.
            com_edge_attr: Optional COM edge attributes ``(E_c, num_rbf)``.
            com_batch: COM -> graph assignment ``(M,)``.

        Returns:
            Updated atom features ``(N, hidden_dim)``.
        """
        # 1. Local atomistic message passing (existing interactions preserved).
        for i, conv in enumerate(self.atom_convs):
            if self.use_residual:
                if edge_attr is not None:
                    h = conv(h, edge_index, edge_attr=edge_attr, batch=batch)
                else:
                    h = conv(h, edge_index, batch=batch)
            else:
                if edge_attr is not None:
                    h = conv(h, edge_index, edge_attr=edge_attr)
                else:
                    h = conv(h, edge_index)
            h = self.act(h)
            if i < self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # 2. Atom -> molecule (permutation-invariant pooling).
        h_com = self.atom_to_com(h, atom_mol, dim_size)  # (M, com_hidden)

        # 3. Molecular (COM) message passing.
        h_com = self.com_mp(h_com, com_edge_index, com_edge_attr, com_batch)

        # 4. Molecule -> atoms (learnable residual).
        return self.com_to_atom(h, h_com, atom_mol)


class HierarchicalMoleculeModel(nn.Module):
    """Two-level (atomistic + COM) graph neural network for molecular regression.

    Shares the configuration surface of
    :class:`~morphology_gnn.model.scaler_model.ScalarMoleculeModel` (``hidden_dim``,
    ``num_layers``, ``conv_class``/``conv_kwargs``, ``use_edge_features``,
    ``pbc_edge_features``, ``num_rbf``/``rbf_kwargs``, cutoffs, ``use_residual``,
    ``residual_kwargs``, ``norm``, ``dropout``, ``global_aggr``, ``num_targets``)
    and adds the hierarchical knobs:

    * ``num_hierarchical_layers`` — number of independent hierarchical cycles
      (each cycle = atomistic stack -> AtomToCOM -> COM message passing ->
      COMToAtom). ``0`` disables the hierarchy entirely and the model falls back
      to the plain atomistic behaviour of ``ScalarMoleculeModel``.
    * ``com_cutoff`` — radius used to build the COM–COM graph from the molecular
      COM positions (also the cutoff of the COM RBF distance embedding).
    * ``com_aggregation`` — ``mean`` (default), ``sum`` or ``attention``.
    * ``com_hidden_channels`` — width of the COM features (default ``hidden_dim``).
    * ``com_num_layers`` — number of COM message-passing layers per cycle.
    * ``com_gated`` — use the gated residual in :class:`COMToAtom` (default True).
    * ``atomistic_edges`` — ``"intra"`` (default) restricts the atomistic graph
      to intra-molecular edges, so molecules communicate *only* through the COM
      level (no atomistic cross-talk); ``"all"`` keeps the original atomistic
      graph (cross-molecule atomistic edges included).
    * ``com_graph`` — ``"radius"`` (default) builds the COM–COM graph from the
      molecular COM distances within ``com_cutoff``; ``"all"`` connects every
      molecule of each sample (a fully-connected COM graph, so all molecules in
      a large vicinity participate in COM–COM interactions).

    The hierarchical path requires a per-node molecule assignment (``mol_number``)
    and node positions ``pos``; when ``mol_number`` is not provided each graph is
    treated as a single molecule (graceful degradation). COM positions are
    computed in-model, mass-weighted and PBC-unwrapped (reusing
    :func:`~morphology_gnn.radius_graph.pbc_center_of_mass`), so the dataset does
    not need to supply them.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_hierarchical_layers: int = 1,
        com_cutoff: float = 6.0,
        com_aggregation: str = "mean",
        com_hidden_channels: int | None = None,
        com_num_layers: int = 1,
        com_gated: bool = True,
        atomistic_edges: str = "intra",
        com_graph: str = "radius",
        conv_class=GATConv,
        conv_kwargs: dict | None = None,
        act=F.gelu,
        dropout: float = 0.2,
        global_aggr=MeanAggregation,
        atom_emb_kwargs: dict | None = None,
        use_edge_features: bool = False,
        pbc_edge_features: bool = False,
        num_rbf: int = 50,
        rbf_kwargs: dict | None = None,
        cutoff_lower: float | None = None,
        cutoff_upper: float | None = None,
        use_residual: bool = True,
        residual_kwargs: dict | None = None,
        norm: str | type[nn.Module] | None = None,
        norm_kwargs: dict | None = None,
        num_targets: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_hierarchical_layers = num_hierarchical_layers
        self.com_cutoff = com_cutoff
        self.com_aggregation = com_aggregation
        self.com_hidden_channels = (
            com_hidden_channels if com_hidden_channels is not None else hidden_dim
        )
        self.com_num_layers = com_num_layers
        self.com_gated = bool(com_gated)
        if atomistic_edges not in ("intra", "all"):
            raise ValueError(
                f"atomistic_edges must be 'intra' or 'all', got {atomistic_edges!r}"
            )
        if com_graph not in ("radius", "all"):
            raise ValueError(f"com_graph must be 'radius' or 'all', got {com_graph!r}")
        self.atomistic_edges = atomistic_edges
        self.com_graph = com_graph
        self.act = act
        self.dropout = dropout
        # `num_hierarchical_layers <= 0` disables the hierarchy -> plain atomistic
        # behaviour (backward compatible with ScalarMoleculeModel).
        self.use_hierarchical = num_hierarchical_layers > 0

        # Global aggregation over nodes -> graph features (same as
        # ScalarMoleculeModel): a single PyG aggregator or several combined via
        # MultiAggregation.
        aggs = build_aggregators(global_aggr)
        self._aggr_out_dim = hidden_dim * len(aggs)
        self.global_aggr = aggs[0] if len(aggs) == 1 else MultiAggregation(aggs)

        self.use_edge_features = use_edge_features
        self.pbc_edge_features = pbc_edge_features
        if pbc_edge_features and not use_edge_features:
            raise ValueError("pbc_edge_features=True requires use_edge_features=True")
        self.num_rbf = num_rbf

        # RBF distance-embedding cutoffs (same first-class handling as
        # ScalarMoleculeModel).
        rbf_kwargs = dict(rbf_kwargs or {})
        rbf_kwargs.setdefault(
            "cutoff_lower", cutoff_lower if cutoff_lower is not None else 0.0
        )
        if cutoff_upper is not None:
            rbf_kwargs.setdefault("cutoff_upper", cutoff_upper)
        self.rbf_kwargs = rbf_kwargs

        # Node features: atomic numbers -> hidden_dim vectors.
        atom_emb_kwargs = dict(atom_emb_kwargs or {})
        atom_emb_kwargs.setdefault("embedding_dim", hidden_dim)
        self.atom_emb: nn.Module = AtomTypeEmbedding(**atom_emb_kwargs)

        # Optional geometric edge attributes (atomistic), reusing EdgeVectorLayer.
        if conv_kwargs is None:
            conv_kwargs = {"heads": 3, "concat": False} if conv_class is GATConv else {}
        else:
            conv_kwargs = dict(conv_kwargs)
        if use_edge_features:
            self.edge_layer = EdgeVectorLayer(num_rbf=num_rbf, rbf_kwargs=rbf_kwargs)
            if "edge_dim" not in inspect.signature(conv_class).parameters:
                raise ValueError(
                    "use_edge_features=True requires a conv class with an "
                    f"`edge_dim` argument (e.g. GATConv, TransformerConv); got "
                    f"{conv_class.__name__}"
                )
            conv_kwargs.setdefault("edge_dim", num_rbf)
        self.conv_kwargs = conv_kwargs

        # Residual / normalization knobs (mirrors ScalarMoleculeModel). The norm
        # spec is resolved separately for the atomistic width and the COM width,
        # since they can differ (com_hidden_channels).
        self.use_residual = use_residual
        residual_kwargs = dict(residual_kwargs or {})
        if norm is not None:
            residual_kwargs.setdefault("norm", norm)
        if norm_kwargs:
            residual_kwargs.setdefault("norm_kwargs", dict(norm_kwargs))
        if dropout is not None:
            residual_kwargs.setdefault("dropout", dropout)
        if not use_residual and (
            norm is not None or dropout is not None or "norm" in residual_kwargs
        ):
            logger.warning(
                "norm/norm_kwargs/dropout are ignored because use_residual=False"
            )
        _norm_spec = residual_kwargs.get("norm")
        _norm_kwargs = dict(residual_kwargs.pop("norm_kwargs", {}) or {})
        residual_kwargs["norm"] = self._resolve_norm_spec(
            _norm_spec, hidden_dim, _norm_kwargs
        )
        self.residual_kwargs = residual_kwargs
        self.com_residual_kwargs: dict | None = None

        if self.use_hierarchical:
            # COM RBF embedding: same basis / envelope as the atomistic one, but
            # with the COM cutoff (used for the COM edge attributes).
            com_rbf_kwargs = dict(rbf_kwargs)
            com_rbf_kwargs["cutoff_upper"] = com_cutoff
            com_rbf_kwargs["cutoff_lower"] = com_rbf_kwargs.get("cutoff_lower", 0.0)
            self.com_residual_kwargs = dict(residual_kwargs)
            self.com_residual_kwargs["norm"] = self._resolve_norm_spec(
                _norm_spec, self.com_hidden_channels, _norm_kwargs
            )
            if use_edge_features:
                self.com_edge_layer = EdgeVectorLayer(
                    num_rbf=num_rbf, rbf_kwargs=com_rbf_kwargs
                )
            self.blocks: nn.ModuleList = nn.ModuleList(
                [
                    HierarchicalBlock(
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        com_hidden_channels=self.com_hidden_channels,
                        com_num_layers=com_num_layers,
                        conv_class=conv_class,
                        conv_kwargs=conv_kwargs,
                        act=act,
                        dropout=dropout,
                        com_aggregation=com_aggregation,
                        use_residual=use_residual,
                        residual_kwargs=residual_kwargs,
                        com_residual_kwargs=self.com_residual_kwargs,
                        gated=com_gated,
                    )
                    for _ in range(num_hierarchical_layers)
                ]
            )
        else:
            # Plain atomistic stack (identical to ScalarMoleculeModel).
            if use_residual:
                self.convs: nn.ModuleList = nn.ModuleList(
                    [
                        Residual(
                            conv_class(hidden_dim, hidden_dim, **conv_kwargs),
                            hidden_dim=hidden_dim,
                            **residual_kwargs,
                        )
                        for _ in range(num_layers)
                    ]
                )
            else:
                self.convs = nn.ModuleList(
                    [
                        conv_class(hidden_dim, hidden_dim, **conv_kwargs)
                        for _ in range(num_layers)
                    ]
                )

        # Graph-level prediction head (same two-layer head as
        # ScalarMoleculeModel): aggregation output -> hidden_dim -> targets.
        self.num_targets = num_targets
        self.lin = nn.Linear(self._aggr_out_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, num_targets)

        # Atomic-number -> atomic-weight (amu) table for the mass-weighted COM
        # positions. Static, so not persisted in checkpoints.
        self._mass_table: torch.Tensor
        self.register_buffer(
            "_mass_table",
            torch.tensor(
                [PT.get_mass(z) for z in range(_MAX_ATOMIC_NUMBER + 1)],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        logger.debug(
            "HierarchicalMoleculeModel(hidden_dim=%d, num_layers=%d, "
            "num_hierarchical_layers=%d, com_cutoff=%.3f, com_aggregation=%s, "
            "com_hidden_channels=%d, com_num_layers=%d)",
            hidden_dim,
            num_layers,
            num_hierarchical_layers,
            com_cutoff,
            com_aggregation,
            self.com_hidden_channels,
            com_num_layers,
        )

    @staticmethod
    def _resolve_norm_spec(norm, dim: int, norm_kwargs: dict | None = None):
        """Resolve a norm spec to something :class:`Residual` can use.

        ``None``/``nn.Identity`` -> ``None`` (Identity), a string name from
        :data:`NORM_REGISTRY` -> a module *instance* sized ``dim`` (matching
        ``ScalarMoleculeModel._resolve_norm``), and a class/instance is passed
        through (``Residual`` instantiates classes with the feature dim).
        """
        norm_kwargs = dict(norm_kwargs or {})
        if norm is None or norm is nn.Identity:
            return None
        if isinstance(norm, nn.Module):
            return norm
        if isinstance(norm, str):
            norm_class = NORM_REGISTRY.get(norm.lower())
            if norm_class is None:
                raise ValueError(
                    f"unknown norm {norm!r}; choose from {sorted(NORM_REGISTRY)}"
                )
            if norm_class is nn.Identity:
                return None
            if norm_class is nn.GroupNorm:
                return norm_class(norm_kwargs.pop("num_groups", 1), dim, **norm_kwargs)
            return norm_class(dim, **norm_kwargs)
        return norm

    def reset_parameters(self) -> None:
        if hasattr(self.atom_emb, "reset_parameters"):
            self.atom_emb.reset_parameters()
        if self.use_hierarchical:
            for block in self.blocks:
                block.reset_parameters()
        else:
            for conv in self.convs:
                conv.reset_parameters()
        self.lin.reset_parameters()
        self.lin2.reset_parameters()

    def _atomistic_edge_attr(
        self,
        pos: torch.Tensor | None,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        box: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Atomistic edge attributes (mirrors ScalarMoleculeModel)."""
        if not self.use_edge_features:
            return None
        if pos is None:
            raise ValueError(
                "use_edge_features=True requires `pos` (node positions) "
                "to be passed to forward"
            )
        if self.pbc_edge_features:
            if box is None:
                raise ValueError(
                    "pbc_edge_features=True requires `box` (per-graph box "
                    "lengths, shape (B, 3)) to be passed to forward"
                )
            box_per_node = box[batch]  # (N, 3)
            return self.edge_layer(pos, edge_index, box_per_node=box_per_node)
        return self.edge_layer(pos, edge_index)

    def _compute_com_positions(
        self,
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        batch: torch.Tensor,
        box: torch.Tensor | None,
        atom_mol: torch.Tensor,
        M: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mass-weighted, PBC-aware COM position of every molecule in the batch.

        Reuses :func:`morphology_gnn.radius_graph.pbc_center_of_mass` per
        molecule: the molecule's atoms are unwrapped across the periodic
        boundary, weighted by their atomic weight (from
        :class:`~morphology_gnn.periodic_table.PT` via a Z -> mass table), and
        the result folded back into the cell. When ``box`` is ``None`` (no PBC)
        a plain mass-weighted mean is used. Only the COM *graph* (relative
        distances) is affected by these positions, so the model stays invariant.

        Args:
            pos: Atom positions ``(N, 3)``.
            atom_types: Atomic numbers ``(N,)`` or ``(N, 1)``.
            batch: Atom -> graph assignment ``(N,)``.
            box: Per-graph box lengths ``(B, 3)`` or ``None``.
            atom_mol: Molecule id per atom ``(N,)`` (``0..M-1``).
            M: Number of COM nodes.

        Returns:
            ``(com_pos (M, 3), com_batch (M,))`` — COM positions and the graph
            index of each COM node.
        """
        device = pos.device
        z = atom_types.reshape(-1)
        masses = self._mass_table[z]  # (N,)
        com_pos = torch.zeros((M, 3), dtype=pos.dtype, device=device)
        # Graph index of each molecule: min over its atoms' batch ids.
        com_batch = scatter(batch, atom_mol, dim=0, dim_size=M, reduce="min")
        for m in range(M):
            mask = atom_mol == m
            n = int(mask.sum().item())
            if n == 0:
                continue
            if box is None:
                msum = masses[mask].sum().clamp_min(1e-12)
                com_pos[m] = (pos[mask] * masses[mask].unsqueeze(-1)).sum(dim=0) / msum
            else:
                g = int(com_batch[m].item())
                com_pos[m] = pbc_center_of_mass(
                    pos[mask], box[g], masses=masses[mask], wrap=True
                )
        return com_pos, com_batch

    def _per_graph_readout(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.global_aggr(h, batch)  # (batch_size, hidden_dim)
        x = self.lin(x)  # (batch_size, hidden_dim)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)  # (batch_size, num_targets)

    def _per_molecule_readout(
        self,
        h: torch.Tensor,
        atom_mol: torch.Tensor,
        M: int,
        mol_is_query: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-molecule readout (same semantics as ScalarMoleculeModel).

        Every molecule of the (query + context) graph is pooled and mapped
        through the head, then only the query molecule(s) per sample are
        returned.
        """
        x = self.global_aggr(h, atom_mol, dim_size=M)  # (M, hidden_dim)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Same two-layer prediction head as the graph-level path.
        x = self.lin(x)  # (M, hidden_dim)
        x = self.act(x)
        pred = self.lin2(x)  # (M, num_targets)
        if mol_is_query is None:
            raise ValueError(
                "mol_is_query is required when mol_number is given (context "
                "mode); mark which atoms belong to the query molecule(s)"
            )
        query_per_mol = (
            scatter(
                mol_is_query.to(pred.device).long(),
                atom_mol,
                dim=0,
                dim_size=M,
                reduce="max",
            )
            > 0
        )  # (M,) bool, one True per sample (the query molecule)
        return pred[query_per_mol]  # (batch_size, num_targets)

    def _forward_atomistic(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        pos: torch.Tensor | None,
        mol_number: torch.Tensor | None,
        mol_is_query: torch.Tensor | None,
        box: torch.Tensor | None,
    ) -> torch.Tensor:
        """Plain atomistic forward — identical to ``ScalarMoleculeModel.forward``."""
        x = self.atom_emb(x)  # (N, hidden_dim)
        edge_attr = self._atomistic_edge_attr(pos, edge_index, batch, box)
        for i, conv in enumerate(self.convs):
            if self.use_residual:
                if edge_attr is not None:
                    x = conv(x, edge_index, edge_attr=edge_attr, batch=batch)
                else:
                    x = conv(x, edge_index, batch=batch)
            else:
                if edge_attr is not None:
                    x = conv(x, edge_index, edge_attr=edge_attr)
                else:
                    x = conv(x, edge_index)
            x = self.act(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if mol_number is not None:
            mol_stride = int(mol_number.max().item()) + 1
            mol_key = mol_number + batch * mol_stride  # unique (sample, molecule)
            x = self.global_aggr(x, mol_key)  # (total_mol, hidden_dim)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Same two-layer prediction head as the graph-level path below.
            x = self.lin(x)  # (total_mol, hidden_dim)
            x = self.act(x)
            pred = self.lin2(x)  # (total_mol, num_targets)
            if mol_is_query is None:
                raise ValueError(
                    "mol_is_query is required when mol_number is given (context "
                    "mode); mark which atoms belong to the query molecule(s)"
                )
            query_per_mol = (
                scatter(
                    mol_is_query.to(pred.device).long(),
                    mol_key,
                    dim=0,
                    reduce="max",
                )
                > 0
            )  # (total_mol,) bool, one True per sample (the query molecule)
            return pred[query_per_mol]  # (batch_size, num_targets)

        x = self.global_aggr(x, batch)  # (batch_size, hidden_dim)
        x = self.lin(x)  # (batch_size, hidden_dim)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)  # (batch_size, num_targets)

    def _forward_hierarchical(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        pos: torch.Tensor | None,
        mol_number: torch.Tensor | None,
        mol_is_query: torch.Tensor | None,
        box: torch.Tensor | None,
    ) -> torch.Tensor:
        """Two-level forward: atoms -> COM -> COM MP -> atoms, repeated."""
        if pos is None:
            raise ValueError(
                "hierarchical mode requires `pos` (node positions) to be "
                "passed to forward (used to compute the COM positions)"
            )
        # Molecular grouping: use the existing per-node `mol_number` when given
        # (box context mode), otherwise treat each graph as a single molecule.
        per_molecule = mol_number is not None
        if mol_number is None:
            mol_number = batch
        mol_number = mol_number.long()
        batch = batch.long()

        # Separate inter-molecular channel: the atomistic graph is restricted to
        # intra-molecular edges, so molecules do NOT exchange information at the
        # atom level; all cross-molecule communication goes through the COM level
        # (COM–COM graph). ``atomistic_edges="all"`` keeps the original graph.
        if self.atomistic_edges == "intra":
            edge_index = filter_intra_molecular_edges(edge_index, mol_number)

        h = self.atom_emb(x)  # (N, hidden_dim)
        edge_attr = self._atomistic_edge_attr(pos, edge_index, batch, box)
        # Opaque, batch-unique grouping key (survives PyG collate offsets).
        mol_stride = int(mol_number.max().item()) + 1
        mol_key = mol_number + batch * mol_stride
        # Contiguous per-(graph, molecule) ids 0..M-1: the *inverse* maps each
        # atom to its molecule, `M` is the number of distinct molecules.
        _mol_unique, atom_mol = torch.unique(mol_key, return_inverse=True)
        M = int(_mol_unique.numel())

        # COM geometry (computed once — the molecular graph is static).
        com_pos, com_batch = self._compute_com_positions(
            pos, x, batch, box, atom_mol, M
        )
        com_edge_index = build_com_graph(
            com_pos, com_batch, box, self.com_cutoff, mode=self.com_graph
        )
        com_edge_attr = None
        if self.use_edge_features:
            com_edge_attr = self.com_edge_layer(
                com_pos,
                com_edge_index,
                box_per_node=box[com_batch] if box is not None else None,
            )

        # Repeated hierarchical cycles (independent blocks).
        for block in self.blocks:
            h = block(
                h,
                edge_index,
                edge_attr,
                batch,
                atom_mol,
                M,
                com_edge_index,
                com_edge_attr,
                com_batch,
            )

        if per_molecule:
            return self._per_molecule_readout(h, atom_mol, M, mol_is_query)
        return self._per_graph_readout(h, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        pos: torch.Tensor | None = None,
        mol_number: torch.Tensor | None = None,
        mol_is_query: torch.Tensor | None = None,
        box: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Graph-level (or per-molecule) scalar predictions.

        Args:
            x: ``(N,)`` atom types.
            edge_index: ``(2, E)`` atomistic connectivity.
            batch: ``(N,)`` graph index per node.
            pos: Optional ``(N, 3)`` node positions (required with
                ``use_edge_features`` and in hierarchical mode).
            mol_number: Optional ``(N,)`` molecule id per node (context mode).
            mol_is_query: Optional ``(N,)`` boolean mask marking the query
                molecule's atoms (context mode).
            box: Optional ``(B, 3)`` per-graph box lengths (required with
                ``pbc_edge_features`` and for the PBC COM path).

        Returns:
            Predictions of shape ``(batch_size, num_targets)``.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "HierarchicalMoleculeModel.forward nodes=%d edges=%d graphs=%d "
                "hierarchical=%s device=%s",
                x.numel(),
                edge_index.shape[1],
                int(batch.max()) + 1 if batch.numel() else 0,
                self.use_hierarchical,
                x.device,
            )
        if not self.use_hierarchical:
            return self._forward_atomistic(
                x, edge_index, batch, pos, mol_number, mol_is_query, box
            )
        return self._forward_hierarchical(
            x, edge_index, batch, pos, mol_number, mol_is_query, box
        )
