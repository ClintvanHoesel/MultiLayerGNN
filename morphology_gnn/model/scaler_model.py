import inspect
import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.nn.aggr import MeanAggregation, MultiAggregation
from torch_geometric.utils import scatter
from .residual import NORM_REGISTRY, Residual

from .embedding import AtomTypeEmbedding, EdgeVectorLayer
from .aggregator import build_aggregators

logger = logging.getLogger(__name__)


class ScalarMoleculeModel(torch.nn.Module):
    """Graph neural network for graph-level scalar regression (e.g. HOMO/LUMO).

    The message-passing layer is pluggable: pass any ``torch_geometric`` conv
    class (``GATConv``, ``GCNConv``, ``SAGEConv``, ``GINConv``, ...) via
    ``conv_class`` and any number of layers via ``num_layers``. Every conv layer
    maps ``hidden_dim -> hidden_dim`` so the width stays constant through the
    stack — for multi-head convs like ``GATConv`` keep ``concat=False`` (the
    default here) so the output width stays ``hidden_dim``.

    Set ``use_edge_features=True`` to compute geometric edge attributes from node
    positions (via :class:`EdgeVectorLayer`) and feed them to the convs as
    ``edge_attr`` — requires a conv class with an ``edge_dim`` argument (e.g.
    ``GATConv``) and passing ``pos`` to ``forward``. The RBF distance embedding
    has configurable cutoffs: ``cutoff_lower`` / ``cutoff_upper`` (first-class
    defaults; entries in ``rbf_kwargs`` take precedence). The runner defaults
    ``cutoff_upper`` to the radius-graph cutoff so the two stay consistent.

    Every conv layer is wrapped in a :class:`Residual` connection by default;
    disable it with ``use_residual=False`` and tune the wrapper via
    ``residual_kwargs``: ``dropout``, ``pre_norm`` / ``post_norm``. The
    normalization type is chosen with ``norm`` (a top-level knob): ``Identity``
    (default), ``LayerNorm``, ``BatchNorm``, ``GraphNorm`` or ``InstanceNorm``
    (from ``torch_geometric.nn.norm``), with extra kwargs in ``norm_kwargs``
    (e.g. ``{"track_running_stats": False}`` for BatchNorm). Norms that support
    per-graph statistics (LayerNorm / GraphNorm / InstanceNorm) automatically
    receive the PyG ``batch`` vector; Identity / BatchNorm do not.

    ``num_targets`` sets the number of graph-level outputs (one per target
    property), so the same model trains on one or several properties at once.

    ``global_aggr`` selects the node-to-graph aggregation: a single PyG
    aggregator (e.g. ``MeanAggregation``), or several combined via
    :class:`MultiAggregation` (pass ``"MeanAggregation+MaxAggregation"`` or a
    list). Any ``torch_geometric.nn.aggr`` name is imported on demand.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
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
        self.act = act
        self.dropout = dropout
        # Global aggregation over nodes -> graph features. Accepts a single
        # aggregator or several (e.g. "MeanAggregation+MaxAggregation" / a list),
        # combined with torch_geometric's MultiAggregation, which concatenates
        # their outputs -> the graph-feature dim scales by the aggregator count.
        aggs = build_aggregators(global_aggr)
        self._aggr_out_dim = hidden_dim * len(aggs)
        self.global_aggr = aggs[0] if len(aggs) == 1 else MultiAggregation(aggs)

        self.use_edge_features = use_edge_features
        self.pbc_edge_features = pbc_edge_features
        if pbc_edge_features and not use_edge_features:
            raise ValueError("pbc_edge_features=True requires use_edge_features=True")
        self.num_rbf = num_rbf

        # RBF distance-embedding cutoffs: `cutoff_lower` / `cutoff_upper` are
        # first-class defaults for the distance embedding; explicit entries in
        # `rbf_kwargs` take precedence (same pattern as num_rbf / rbf_kwargs).
        # `cutoff_upper=None` keeps the DistanceEmbedding default (5.0 Å) unless
        # overridden via rbf_kwargs or by the runner (which defaults it to the
        # radius-graph cutoff).
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
        self.atom_emb = AtomTypeEmbedding(**atom_emb_kwargs)

        # Optional geometric edge attributes: neighbour displacements through
        # the RBF distance embedding, consumed by the convs as edge_attr.
        if conv_kwargs is None:
            conv_kwargs = {"heads": 3, "concat": False} if conv_class is GATConv else {}
        else:
            conv_kwargs = dict(conv_kwargs)
        if use_edge_features:
            self.edge_layer = EdgeVectorLayer(num_rbf=num_rbf, rbf_kwargs=rbf_kwargs)
            logger.debug(
                "edge features: num_rbf=%d rbf_cutoff=(%s, %s)",
                num_rbf,
                self.rbf_kwargs.get("cutoff_lower"),
                self.rbf_kwargs.get("cutoff_upper"),
            )
            if "edge_dim" not in inspect.signature(conv_class).parameters:
                raise ValueError(
                    "use_edge_features=True requires a conv class with an "
                    f"`edge_dim` argument (e.g. GATConv, TransformerConv); got "
                    f"{conv_class.__name__}"
                )
            conv_kwargs.setdefault("edge_dim", num_rbf)

        # Stack of num_layers message-passing layers (any conv class), each
        # optionally wrapped in a Residual connection. The wrapper is
        # configurable via `residual_kwargs` (dropout, pre/post norm, norm
        # type); set `use_residual=False` to stack plain convs.
        self.use_residual = use_residual
        residual_kwargs = dict(residual_kwargs or {})
        # First-class normalization knob: `norm` / `norm_kwargs` feed the
        # residual wrapper unless it already sets `norm` (more specific).
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
        self._resolve_norm(residual_kwargs, hidden_dim)
        self.residual_kwargs = residual_kwargs
        logger.debug(
            "ScalarMoleculeModel(hidden_dim=%d, num_layers=%d, conv=%s, "
            "use_residual=%s, norm=%s)",
            hidden_dim,
            num_layers,
            conv_class.__name__,
            use_residual,
            (
                type(self.residual_kwargs.get("norm")).__name__
                if self.residual_kwargs.get("norm") is not None
                else "Identity"
            ),
        )
        if use_residual:
            self.convs = nn.ModuleList(
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

        # Graph-level prediction head: one output per target property. The input
        # dim follows the (possibly multi) aggregation output width.
        self.num_targets = num_targets
        self.lin = nn.Linear(self._aggr_out_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, num_targets)

    @staticmethod
    def _resolve_norm(residual_kwargs: dict, hidden_dim: int) -> None:
        """Turn a string ``norm`` name into a norm module (in place).

        ``residual_kwargs`` is mutated: ``norm`` becomes a module instance (or
        ``None`` for Identity) and the ``norm_kwargs`` key (not a ``Residual``
        argument) is consumed. Names come from :data:`NORM_REGISTRY`.
        """
        norm_kwargs = dict(residual_kwargs.pop("norm_kwargs", {}) or {})
        norm = residual_kwargs.get("norm")
        if not isinstance(norm, str):
            return
        norm_class = NORM_REGISTRY.get(norm.lower())
        if norm_class is None:
            raise ValueError(
                f"unknown norm {norm!r}; choose from {sorted(NORM_REGISTRY)}"
            )
        logger.debug("resolved norm %r -> %s", norm, norm_class.__name__)
        if norm_class is nn.Identity:
            residual_kwargs["norm"] = None
        elif norm_class is nn.GroupNorm:
            residual_kwargs["norm"] = norm_class(
                norm_kwargs.pop("num_groups", 1), hidden_dim, **norm_kwargs
            )
        else:
            residual_kwargs["norm"] = norm_class(hidden_dim, **norm_kwargs)

    def reset_parameters(self) -> None:
        if hasattr(self.atom_emb, "reset_parameters"):
            self.atom_emb.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()
        self.lin2.reset_parameters()

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
            edge_index: ``(2, E)`` connectivity.
            batch: ``(N,)`` graph index per node.
            pos: Optional ``(N, 3)`` node positions (required with
                ``use_edge_features``).
            mol_number: Optional ``(N,)`` molecule id per node (query = 0). When
                given, the model runs a *per-molecule* readout over every
                molecule in the (query + context) graph and returns only the
                query molecule's prediction per sample -- surrounding molecules
                are passed through message passing but never trained on.
            mol_is_query: Optional ``(N,)`` boolean mask, True for the query
                molecule's atoms. Required together with ``mol_number`` so the
                query prediction can be extracted from the per-molecule output.
            box: Optional ``(B, 3)`` per-graph box lengths. Required with
                ``pbc_edge_features`` (minimum-image edge displacements).

        Returns:
            Predictions of shape ``(batch_size, num_targets)``.
        """
        # x: (N,) atom types; edge_index: (2, E); batch: (N,)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "ScalarMoleculeModel.forward nodes=%d edges=%d graphs=%d device=%s",
                x.numel(),
                edge_index.shape[1],
                int(batch.max()) + 1 if batch.numel() else 0,
                x.device,
            )
        x = self.atom_emb(x)  # (N, hidden_dim)
        edge_attr = None
        if self.use_edge_features:
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
                edge_attr = self.edge_layer(
                    pos, edge_index, box_per_node=box_per_node
                )  # (E, num_rbf)
            else:
                edge_attr = self.edge_layer(pos, edge_index)  # (E, num_rbf)
        for i, conv in enumerate(self.convs):
            if self.use_residual:
                # The residual wrapper consumes `batch` for per-graph
                # normalization (it never reaches the conv sublayer).
                if edge_attr is not None:
                    x = conv(x, edge_index, edge_attr=edge_attr, batch=batch)
                else:
                    x = conv(x, edge_index, batch=batch)
            else:
                # Plain convs do not accept a `batch` argument.
                if edge_attr is not None:
                    x = conv(x, edge_index, edge_attr=edge_attr)
                else:
                    x = conv(x, edge_index)
            x = self.act(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if mol_number is not None:
            # Per-molecule readout over the whole (query + context) graph: every
            # molecule of the graph is pooled to one vector and mapped through
            # the head, then only the query molecule's row is returned so the
            # loss is computed on the minibatch (query) molecules only.
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
