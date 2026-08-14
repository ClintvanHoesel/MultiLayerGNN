"""SE(3)-equivariant denoising diffusion model for molecular positions.

Learns to denoise atomic coordinates — a molecule conformation inside a
periodic cell — conditioned on the atom types, the cell, and a noise level
``t in [0, 1]``. The network is PBC-aware: edge features use minimum-image
displacements and the radius graph is rebuilt from the (noisy) coordinates at
every step by the caller (see :mod:`morphology_gnn.model.diffusion_trainer`).

Trained with an epsilon-parameterized DDPM (cosine schedule by default): the
forward process is ``x_t = x_0 + sigma(t) * eps`` with ``eps ~ N(0, I)`` and the
model predicts ``eps`` from ``x_t``. Equivariance is exact SE(3) via an
EGNN-style directional noise head (no spherical harmonics / Wigner-D): the GNN
backbone produces an *invariant* hidden state ``h`` and the head combines it
with minimum-image unit vectors, so the predicted noise field rotates with the
coordinates and is translation-invariant.

Reuses the existing stack: :class:`AtomTypeEmbedding`, :class:`Residual` +
``NORM_REGISTRY``, PyG convs (``GATConv`` default) and the RBF distance
embedding. No new dependencies.
"""

from __future__ import annotations

import inspect
import logging
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv

from .embedding import AtomTypeEmbedding, DistanceVectors, ScalarDistanceEmbedding
from .residual import NORM_REGISTRY, Residual

logger = logging.getLogger(__name__)


def resolve_norm_kwargs(residual_kwargs: dict, hidden_dim: int) -> None:
    """Turn a string ``norm`` name into a norm module (in place).

    Same behavior as ``ScalarMoleculeModel._resolve_norm`` (kept local so the
    diffusion model does not depend on the scalar-regression model): ``norm``
    becomes a module instance (or ``None`` for Identity) and the ``norm_kwargs``
    key (not a ``Residual`` argument) is consumed.
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


def min_image_disp_batched(
    pos: torch.Tensor, edge_index: torch.Tensor, box_per_node: torch.Tensor
) -> torch.Tensor:
    """Minimum-image displacement vectors for batched edges (orthorhombic cells).

    Args:
        pos: Node positions, shape ``(N, 3)`` (concatenated over graphs).
        edge_index: Connectivity, shape ``(2, E)``.
        box_per_node: Per-node box lengths, shape ``(N, 3)`` (each node carries
            its graph's box). Edges never cross graphs, so the box of the edge
            source applies to the whole edge.

    Returns:
        Displacement vectors of shape ``(E, 3)`` (``pos[dst] - pos[src]`` under
        the minimum-image convention).
    """
    src, dst = edge_index[0], edge_index[1]
    disp = pos[dst] - pos[src]  # (E, 3)
    box_e = box_per_node[src]  # (E, 3)
    return disp - torch.round(disp / box_e) * box_e


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding (``t in [0, 1]``) mapped through an MLP.

    Standard DDPM positional time embedding: ``sin/cos`` of ``t`` at geometrically
    spaced frequencies, then a small MLP to ``out_dim``.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        out_dim: int | None = None,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        out_dim = out_dim or hidden_dim
        self.hidden_dim = hidden_dim
        self.base = base
        half = max(hidden_dim // 2, 1)
        self.register_buffer(
            "freqs",
            torch.exp(-torch.arange(half).float() * math.log(base) / half),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Map times (any shape) to embeddings of shape ``(*shape, out_dim)``."""
        args = t.unsqueeze(-1) * self.freqs  # (..., half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (..., hidden)
        return self.mlp(emb)


class PBCScalarDistanceEmbedding(nn.Module):
    """RBF-embed minimum-image inter-atomic distances (PBC-aware edge features).

    Unlike :class:`EdgeVectorLayer` (which uses the raw ``pos[dst] - pos[src]``
    displacement — wrong under PBC once noise pushes atoms across the box), this
    computes the minimum-image displacement before embedding the distance.
    """

    def __init__(self, num_rbf: int = 50, rbf_kwargs: dict | None = None) -> None:
        super().__init__()
        self.edge_emb = ScalarDistanceEmbedding(
            num_rbf=num_rbf, rbf_kwargs=rbf_kwargs
        )

    @property
    def embedding_dim(self) -> int:
        return self.edge_emb.embedding_dim

    def forward(
        self, pos: torch.Tensor, edge_index: torch.Tensor, box_per_node: torch.Tensor
    ) -> torch.Tensor:
        """Embed edge displacements to RBF features of shape ``(E, num_rbf)``."""
        disp = min_image_disp_batched(pos, edge_index, box_per_node)
        return self.edge_emb(disp)


class EquivariantNoiseHead(nn.Module):
    """EGNN-style SE(3)-equivariant noise head.

    Computes::

        eps_i = sum_{j in N(i)} phi([h_i || h_j || rbf_ij]) * rhat_ij

    where ``h`` is the (invariant) GNN hidden state, ``rbf_ij`` the RBF-embedded
    minimum-image distance and ``rhat_ij`` the minimum-image unit vector to
    neighbor ``j``. Because ``phi`` only sees rotation-invariant quantities and
    the messages are the (equivariant) unit vectors, the output field ``eps``
    rotates with the coordinates and is translation-invariant (exact SE(3),
    without spherical harmonics).

    An optional isotropic self term ``psi(h_i)`` (per-atom, invariant) is
    initialized to zero so the head starts predicting ~zero noise.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_rbf: int,
        hidden: int | None = None,
        self_term: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.hidden = hidden or hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_rbf, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, 1),
        )
        # Zero-initialize the head output so the network starts predicting ~0
        # noise (the mean of the Gaussian target) — well-conditioned start.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.self_term = self_term
        if self_term:
            self.self_mlp = nn.Sequential(
                nn.Linear(hidden_dim, self.hidden),
                nn.SiLU(),
                nn.Linear(self.hidden, 3),
            )
            nn.init.zeros_(self.self_mlp[-1].weight)
            nn.init.zeros_(self.self_mlp[-1].bias)

    def reset_parameters(self) -> None:
        for mod in self.mlp:
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
        if self.self_term:
            nn.init.zeros_(self.self_mlp[-1].weight)
            nn.init.zeros_(self.self_mlp[-1].bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        rbf: torch.Tensor,
        r_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the per-atom noise field.

        Args:
            h: Invariant hidden state, shape ``(N, hidden_dim)``.
            edge_index: Connectivity, shape ``(2, E)``.
            rbf: RBF edge features, shape ``(E, num_rbf)``.
            r_hat: Minimum-image unit displacement per edge, shape ``(E, 3)``.

        Returns:
            Predicted noise ``eps`` of shape ``(N, 3)``.
        """
        src, dst = edge_index[0], edge_index[1]
        mlp_in = torch.cat([h[src], h[dst], rbf], dim=-1)  # (E, 2*hidden + num_rbf)
        w = self.mlp(mlp_in)  # (E, 1)
        msg = w * r_hat  # (E, 3)
        eps = torch.index_add(
            torch.zeros(h.shape[0], 3, device=h.device, dtype=h.dtype),
            0,
            src,
            msg,
        )
        if self.self_term:
            eps = eps + self.self_mlp(h)
        return eps


class NoiseSchedule(nn.Module):
    """Continuous-time noise schedule (epsilon-parameterization).

    ``alpha_bar(t)`` is the retained-signal fraction at ``t in [0, 1]`` and
    ``sigma(t) = sqrt(1 - alpha_bar(t))`` the injected-noise standard deviation,
    so the forward process is ``x_t = x_0 + sigma(t) * eps`` with
    ``eps ~ N(0, I)``. ``sigma(0) = 0`` (clean data) and ``sigma(1) = 1`` (pure
    noise), with monotone interpolation in between.
    """

    SCHEDULES = ("cosine", "linear")

    def __init__(self, kind: str = "cosine") -> None:
        super().__init__()
        if kind not in self.SCHEDULES:
            raise ValueError(
                f"unknown noise schedule {kind!r}; choose from {self.SCHEDULES}"
            )
        self.kind = kind

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Retained-signal fraction at time ``t`` (any shape, in ``[0, 1]``)."""
        if self.kind == "cosine":
            return (torch.cos(0.5 * math.pi * t) ** 2).clamp_min(0.0)
        # linear: sigma(t)^2 = t -> alpha_bar(t) = 1 - t
        return (1.0 - t).clamp_min(0.0)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise standard deviation at time ``t``."""
        return torch.sqrt((1.0 - self.alpha_bar(t)).clamp_min(0.0))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio ``alpha_bar / (1 - alpha_bar)``."""
        return self.alpha_bar(t) / (1.0 - self.alpha_bar(t)).clamp_min(1e-8)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma(t)


class DiffusionMoleculeModel(nn.Module):
    """SE(3)-equivariant denoiser for molecular coordinates.

    Predicts the noise ``eps`` of the forward diffusion process
    ``x_t = x_0 + sigma(t) * eps`` given the noisy (in-cell) coordinates, atom
    types, cell and noise level.

    Args:
        hidden_dim: Width of the node features / hidden state.
        num_layers: Number of message-passing layers.
        conv_class: Any ``torch_geometric`` conv class that accepts an
            ``edge_dim`` argument (default ``GATConv``, which consumes the RBF
            edge features).
        conv_kwargs: Extra kwargs for every conv layer (e.g.
            ``{"heads": 4, "concat": False}``).
        act: Activation applied after each conv layer.
        dropout: Dropout between conv layers.
        num_rbf: Number of RBF basis functions for the min-image edge distances.
        rbf_kwargs: Deep kwargs for the RBF distance embedding (e.g.
            ``{"rbf_class": ExpNormalSmearing}``).
        cutoff_lower / cutoff_upper: RBF distance cutoffs (first-class defaults;
            explicit ``rbf_kwargs`` entries win). ``cutoff_upper=None`` keeps
            the distance-embedding default unless the runner defaults it to the
            radius-graph cutoff.
        use_residual: Wrap each conv in a :class:`Residual` connection.
        residual_kwargs: Deep kwargs for the residual wrapper (``dropout``,
            ``pre_norm``, ``post_norm``).
        norm / norm_kwargs: First-class normalization knob (names from
            :data:`NORM_REGISTRY`), applied inside the residual wrapper.
        cell_embed_dim: Width of the MLP that embeds the (3,) box lengths.
        self_term: Whether the noise head includes an isotropic per-atom term.
        noise_schedule: ``"cosine"`` (default) or ``"linear"``.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        conv_class=GATConv,
        conv_kwargs: dict | None = None,
        act=F.silu,
        dropout: float = 0.1,
        num_rbf: int = 50,
        rbf_kwargs: dict | None = None,
        cutoff_lower: float | None = None,
        cutoff_upper: float | None = None,
        use_residual: bool = True,
        residual_kwargs: dict | None = None,
        norm: str | type[nn.Module] | None = None,
        norm_kwargs: dict | None = None,
        cell_embed_dim: int = 16,
        self_term: bool = True,
        noise_schedule: str = "cosine",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.act = act
        self.dropout = dropout
        self.cell_embed_dim = cell_embed_dim

        # RBF cutoffs: `cutoff_lower`/`cutoff_upper` are first-class defaults;
        # explicit `rbf_kwargs` entries take precedence.
        rbf_kwargs = dict(rbf_kwargs or {})
        rbf_kwargs.setdefault(
            "cutoff_lower", cutoff_lower if cutoff_lower is not None else 0.0
        )
        if cutoff_upper is not None:
            rbf_kwargs.setdefault("cutoff_upper", cutoff_upper)
        self.rbf_kwargs = rbf_kwargs

        # Node features: atom types + time + cell (box lengths), all -> hidden.
        self.atom_emb = AtomTypeEmbedding(embedding_dim=hidden_dim)
        self.time_emb = TimeEmbedding(hidden_dim, hidden_dim)
        self.cell_emb = nn.Sequential(
            nn.Linear(3, cell_embed_dim),
            nn.SiLU(),
            nn.Linear(cell_embed_dim, hidden_dim),
        )
        # Edge features: min-image distances through RBFs.
        self.edge_emb = PBCScalarDistanceEmbedding(
            num_rbf=num_rbf, rbf_kwargs=rbf_kwargs
        )

        # Message-passing backbone (mirrors ScalarMoleculeModel). The conv must
        # accept `edge_dim` because we always feed RBF edge features.
        conv_kwargs = dict(conv_kwargs or {})
        if conv_class is GATConv:
            conv_kwargs.setdefault("heads", 4)
            conv_kwargs.setdefault("concat", False)
        if "edge_dim" not in inspect.signature(conv_class).parameters:
            raise ValueError(
                f"the diffusion backbone needs a conv class with an `edge_dim` "
                f"argument (e.g. GATConv, TransformerConv); got {conv_class.__name__}"
            )
        conv_kwargs.setdefault("edge_dim", num_rbf)
        self.conv_class = conv_class

        self.use_residual = use_residual
        residual_kwargs = dict(residual_kwargs or {})
        if norm is not None:
            residual_kwargs.setdefault("norm", norm)
        if norm_kwargs:
            residual_kwargs.setdefault("norm_kwargs", dict(norm_kwargs))
        resolve_norm_kwargs(residual_kwargs, hidden_dim)
        self.residual_kwargs = residual_kwargs
        logger.debug(
            "DiffusionMoleculeModel(hidden_dim=%d, num_layers=%d, conv=%s, "
            "num_rbf=%d, use_residual=%s, norm=%s, schedule=%s)",
            hidden_dim,
            num_layers,
            conv_class.__name__,
            num_rbf,
            use_residual,
            (
                type(residual_kwargs.get("norm")).__name__
                if residual_kwargs.get("norm") is not None
                else "Identity"
            ),
            noise_schedule,
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

        self.noise_head = EquivariantNoiseHead(
            hidden_dim, num_rbf, hidden=hidden_dim, self_term=self_term
        )
        self.noise_schedule = NoiseSchedule(kind=noise_schedule)

    def reset_parameters(self) -> None:
        for mod in (
            self.atom_emb,
            self.time_emb,
            self.cell_emb,
            self.edge_emb,
            self.noise_head,
        ):
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        pos_noisy: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        t: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise field.

        Args:
            x: Atom types, shape ``(N,)`` or ``(N, 1)`` (atomic numbers).
            pos_noisy: Noisy in-cell positions, shape ``(N, 3)``.
            edge_index: Connectivity rebuilt from ``pos_noisy``, shape ``(2, E)``.
            batch: Node -> graph assignment, shape ``(N,)``.
            t: Per-graph timesteps in ``[0, 1]``, shape ``(B,)``.
            box: Per-graph box lengths, shape ``(B, 3)``.

        Returns:
            Predicted noise ``eps`` of shape ``(N, 3)``.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "DiffusionMoleculeModel.forward nodes=%d edges=%d graphs=%d",
                pos_noisy.shape[0],
                edge_index.shape[1],
                int(batch.max()) + 1 if batch.numel() else 0,
            )
        t_node = t[batch]  # (N,)
        box_node = box[batch]  # (N, 3)
        h = self.atom_emb(x) + self.time_emb(t_node) + self.cell_emb(box_node)

        edge_attr = self.edge_emb(pos_noisy, edge_index, box_node)  # (E, num_rbf)
        for i, conv in enumerate(self.convs):
            if self.use_residual:
                # The residual wrapper consumes `batch` for per-graph norms; it
                # never reaches the conv sublayer.
                h = conv(h, edge_index, edge_attr=edge_attr, batch=batch)
            else:
                h = conv(h, edge_index, edge_attr=edge_attr)
            h = self.act(h)
            if i < self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        disp = min_image_disp_batched(pos_noisy, edge_index, box_node)  # (E, 3)
        r_hat, _ = DistanceVectors()(disp)  # (E, 3) unit min-image displacement
        return self.noise_head(h, edge_index, edge_attr, r_hat)  # (N, 3)
