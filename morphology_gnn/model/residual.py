import inspect
import logging

import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable
import torch_geometric
import torch_geometric.nn

logger = logging.getLogger(__name__)

NORM_REGISTRY = {
    "identity": nn.Identity,
    "layernorm": torch_geometric.nn.norm.LayerNorm,
    "batchnorm": torch_geometric.nn.norm.BatchNorm,
    "graphnorm": torch_geometric.nn.norm.GraphNorm,
    "instancenorm": torch_geometric.nn.norm.InstanceNorm,
}


def _resolve_sublayer_dim(sublayer: nn.Module) -> int | None:
    """Best-effort feature dimension of a wrapped sublayer."""
    for attr in ("hidden_dim", "out_channels", "in_channels"):
        value = getattr(sublayer, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


class Residual(nn.Module):
    """Wrap a sublayer with a pre- or post-normalization residual connection.

    Works for any sublayer, including GNN message-passing layers that need extra
    arguments such as ``edge_index``: extra positional/keyword args passed to
    ``forward`` are forwarded to the wrapped sublayer::

        x = Residual(GATConv(d, d), d)(x, edge_index)

    The wrapped module is exposed as ``.sublayer``.

    Args:
        sublayer: The module to wrap (e.g. a ``torch_geometric`` conv).
        dropout: Dropout applied to the sublayer output before the addition.
        pre_norm: Apply ``norm`` to ``x`` *before* the sublayer.
        post_norm: Apply ``norm`` to the sum *after* the residual addition.
        norm: ``None`` / ``nn.Identity`` (no normalization), an ``nn.Module``
            *instance*, or a module *class* instantiated with the feature
            dimension. ``nn.GroupNorm`` is special-cased as ``GroupNorm(1, dim)``
            (instance-norm-like). See :data:`NORM_REGISTRY` for the names the
            config layer accepts.
        hidden_dim: Feature dimension of the sublayer, used to build ``norm``.
            When omitted it is inferred from the sublayer (``hidden_dim``,
            ``out_channels``, then ``in_channels``).

    The PyG ``batch`` vector (node -> graph assignment) can be passed to
    ``forward`` as a keyword argument. Norms that support per-graph statistics
    (``LayerNorm``, ``GraphNorm``, ``InstanceNorm``) receive it; others
    (``Identity``, and PyG ``BatchNorm`` in this version) are called with ``x``
    only. ``batch`` never reaches the wrapped sublayer.
    """

    def __init__(
        self,
        sublayer: nn.Module,
        dropout: float = 0.0,
        pre_norm: bool = False,
        post_norm: bool = False,
        norm: nn.Module | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        # Explicit annotations keep the type checker from inferring these
        # wrapped modules as ``torch.Tensor`` (which would otherwise report
        # "Object of type 'Tensor' is not callable" at the call sites).
        self.sublayer: nn.Module = sublayer
        self.norm: nn.Module
        if norm is None or norm is nn.Identity:
            self.norm = nn.Identity()
        elif isinstance(norm, nn.Module):
            self.norm = norm
        else:
            dim = hidden_dim or _resolve_sublayer_dim(sublayer)
            if dim is None:
                raise ValueError(
                    "cannot infer the feature dimension to build the norm; "
                    "pass `hidden_dim` or provide a norm instance"
                )
            self.norm = nn.GroupNorm(1, dim) if norm is nn.GroupNorm else norm(dim)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        # Some norms (PyG graph norms like LayerNorm / GraphNorm / InstanceNorm)
        # take a `batch` vector for per-graph statistics; detect it so we know
        # whether to pass `batch` at call time. (PyG BatchNorm here does not.)
        self._norm_accepts_batch = (
            "batch" in inspect.signature(self.norm.forward).parameters
        )

    def _apply_norm(self, x: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        """Apply ``self.norm`` to ``x``, forwarding the PyG ``batch`` vector when
        the norm supports it (per-graph statistics)."""
        if batch is not None and self._norm_accepts_batch:
            return self.norm(x, batch)
        return self.norm(x)

    def reset_parameters(self) -> None:
        if hasattr(self.sublayer, "reset_parameters"):
            self.sublayer.reset_parameters()
        if hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # `batch` (PyG node->graph assignment) is consumed by the norm for
        # per-graph statistics and never forwarded to the sublayer (conv).
        batch = kwargs.pop("batch", None)
        logger.debug(
            "Residual.forward(norm=%s, pre_norm=%s, post_norm=%s, batch=%s)",
            type(self.norm).__name__,
            self.pre_norm,
            self.post_norm,
            batch is not None,
        )
        if self.pre_norm:
            x = self._apply_norm(x, batch)
        h = self.dropout(self.sublayer(x, *args, **kwargs))
        h = x + h
        if self.post_norm:
            h = self._apply_norm(h, batch)
        return h


class FeedForward(nn.Module):
    """Position-wise MLP: Linear -> GELU -> Dropout -> Linear."""

    def __init__(
        self,
        hidden_dim: int,
        expansion: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim * expansion)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HistoryAttention(nn.Module):
    """Softmax attention over the layer history (the residual stream).

    Instead of combining the outputs of all previous layers with a plain sum,
    this computes attention weights over the history and multiplies them with
    the (value-projected) history, following the LLM ``softmax(QK^T/\\sqrt{d}) V``
    pattern but across the *layer* dimension::

        e_l  = (W_q q) . (W_k h_l) / sqrt(d)
        a_l  = softmax_l(e)                  # weights over history entries
        out  = sum_l a_l (W_v h_l)           # softmax multiplied with the history

    ``q`` is the query state — the latest layer output — and ``h_l`` are the
    outputs of every previous layer. Because the weights form a softmax, ``out``
    is a convex combination of the history (weights sum to 1), so magnitudes
    stay bounded instead of growing with depth like an unnormalized sum.

    This is *across*-layer (temporal) attention. The *within*-graph (spatial)
    attention over a node's neighbours — with its own softmax — lives inside the
    GNN conv passed to :class:`FullAttentionBlock`.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float | Callable[[torch.Tensor], torch.Tensor] = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.invsqrt_hiddendim = self.hidden_dim ** (-0.5)
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if isinstance(dropout, (float, int)):
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout

    def forward(self, query: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        # history: (L, N, d) — L layers, N nodes, d hidden; query: (N, d)
        q = self.w_q(query).unsqueeze(0)  # (1, N, d)
        k = self.w_k(history)  # (L, N, d)
        v = self.w_v(history)  # (L, N, d)
        scores = (q * k).sum(-1) * self.invsqrt_hiddendim  # (L, N)
        weights = F.softmax(scores, dim=0)  # (L, N)
        weights = self.dropout(weights)
        return (weights.unsqueeze(-1) * v).sum(dim=0)  # (N, d)


class HistoryBlock(nn.Module):
    """Full-attention residual block in the spirit of LLM residual streams.

    The block keeps track of the *history* — the output of every layer processed
    so far — and each new layer reads that history with **softmax attention**
    (softmax weights multiplied with the history) rather than a plain sum, then
    applies spatial GNN attention (message passing) plus a feed-forward network,
    and appends its own output back into the history. This is the "residual
    stream" view of Transformers: every layer reads from and writes to a single
    shared stream, so information flows freely from layer 0 to layer N.

    Stack layers by calling the module once per layer on the same graph::

        block = FullAttentionBlock(hidden_dim=16, attention=GATConv(16, 16))
        for _ in range(num_layers):
            x = block(x, edge_index)
        block.reset()  # clear the history when moving to a new graph

    Any graph structure (``edge_index``, ``batch``, ``pos``, ...) can be passed
    through as extra args — they are forwarded to the attention sublayer.

    .. note::
        The history grows by one tensor per layer, and :class:`HistoryAttention`
        needs the full history (not just a running sum) to attend over it.
    """

    def __init__(
        self,
        hidden_dim: int,
        attention: nn.Module,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = attention
        self.history_attn = HistoryAttention(hidden_dim, dropout=dropout)
        # History of every layer's output. Persisted across calls so the module
        # can be invoked once per layer; call reset() for a new graph.
        self._history: list[torch.Tensor] = []

    def reset(self) -> None:
        """Clear the layer history (call when switching to a new graph)."""
        self._history.clear()

    @property
    def history(self) -> list[torch.Tensor]:
        """Outputs of every layer processed so far, in order."""
        return self._history

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # First call seeds the history with the input embeddings.
        if not self._history:
            self._history.append(x)
        logger.debug("HistoryBlock.forward history length=%d", len(self._history))
        # Full attention: combine the outputs of EVERY previous layer using
        # softmax weights (softmax multiplied with the history), then run the
        # spatial GNN attention + feed-forward on the result.
        history = torch.stack(self._history)  # (L, N, d)
        out = self.history_attn(x, history)  # query = latest layer output
        return out
