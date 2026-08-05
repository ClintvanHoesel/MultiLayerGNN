import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable


class Residual(nn.Module):
    """Wrap a sublayer with a pre- or post-LayerNorm residual connection.

    Works for any sublayer, including GNN message-passing layers that need extra
    arguments such as ``edge_index``: extra positional/keyword args passed to
    ``forward`` are forwarded to the wrapped sublayer::

        x = Residual(GATConv(d, d), d)(x, edge_index)

    The wrapped module is exposed as ``.sublayer``.
    """

    def __init__(
        self,
        sublayer: nn.Module,
        hidden_dim: int,
        dropout: float = 0.0,
        pre_norm: bool = False,
        post_norm: bool = False,
        norm: nn.Module = torch.nn.Identity,
    ) -> None:
        super().__init__()
        self.sublayer = sublayer
        # `norm` is a module *class*; instantiate it (Identity takes no args).
        if isinstance(norm, nn.Module):
            self.norm = norm
        elif norm is torch.nn.Identity:
            self.norm = norm()
        else:
            self.norm = norm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        self.post_norm = post_norm

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.pre_norm:
            x = self.norm(x)
        h = self.dropout(self.sublayer(x, *args, **kwargs))
        h = x + h
        if self.post_norm:
            h = self.norm(h)
        return h


class AttentionResidual(Residual):
    """A residual wrapper specialized for attention / GNN sublayers.

    The wrapped module stays reachable as ``.sublayer``, so for example
    ``block.attention.sublayer.lin_l`` works for a wrapped ``GATConv``.
    """

    def __init__(
        self,
        attention: nn.Module,
        hidden_dim: int,
        dropout: float = 0.0,
        pre_norm: bool = True,
        post_norm: bool = False,
        norm: nn.Module = torch.nn.Identity,
    ) -> None:
        super().__init__(
            attention,
            hidden_dim,
            dropout=dropout,
            pre_norm=pre_norm,
            post_norm=post_norm,
            norm=norm,
        )


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
    the (value-projected) history, following the LLM ``softmax(QK^T/\sqrt{d}) V``
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
        if isinstance(dropout, float):
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
        return (weights.unsqueeze(-1) * history).sum(dim=0)  # (N, d)


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
        # Full attention: combine the outputs of EVERY previous layer using
        # softmax weights (softmax multiplied with the history), then run the
        # spatial GNN attention + feed-forward on the result.
        history = torch.stack(self._history)  # (L, N, d)
        out = self.history_attn(x, history)  # query = latest layer output
        return out
