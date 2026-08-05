import inspect

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.nn.aggr import MeanAggregation

from .embedding import AtomTypeEmbedding, EdgeVectorLayer


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
    ``GATConv``) and passing ``pos`` to ``forward``.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        conv_class=GATConv,
        conv_kwargs: dict | None = None,
        act=F.gelu,
        dropout: float = 0.2,
        global_agrr=MeanAggregation,
        atom_emb_kwargs: dict | None = None,
        use_edge_features: bool = False,
        num_rbf: int = 50,
        rbf_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.act = act
        self.dropout = dropout
        # `global_agrr` is a torch_geometric Aggregation *class* (e.g.
        # MeanAggregation); instantiate it (or accept an already-built instance).
        self.global_agrr = (
            global_agrr() if inspect.isclass(global_agrr) else global_agrr
        )

        self.use_edge_features = use_edge_features
        self.num_rbf = num_rbf

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
            if "edge_dim" not in inspect.signature(conv_class).parameters:
                raise ValueError(
                    "use_edge_features=True requires a conv class with an "
                    f"`edge_dim` argument (e.g. GATConv, TransformerConv); got "
                    f"{conv_class.__name__}"
                )
            conv_kwargs.setdefault("edge_dim", num_rbf)

        # Stack of num_layers message-passing layers (any conv class).
        self.convs = nn.ModuleList(
            [
                conv_class(hidden_dim, hidden_dim, **conv_kwargs)
                for _ in range(num_layers)
            ]
        )

        # Graph-level prediction head.
        self.lin = nn.Linear(hidden_dim, 1)

    def reset_parameters(self) -> None:
        if hasattr(self.atom_emb, "reset_parameters"):
            self.atom_emb.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (N,) atom types; edge_index: (2, E); batch: (N,)
        x = self.atom_emb(x)  # (N, hidden_dim)
        edge_attr = None
        if self.use_edge_features:
            if pos is None:
                raise ValueError(
                    "use_edge_features=True requires `pos` (node positions) "
                    "to be passed to forward"
                )
            edge_attr = self.edge_layer(pos, edge_index)  # (E, num_rbf)
        for i, conv in enumerate(self.convs):
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = self.act(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.global_agrr(x, batch)  # (batch_size, hidden_dim)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)  # (batch_size, 1)
