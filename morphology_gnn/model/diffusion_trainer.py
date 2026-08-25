"""Lightning trainer, sampler and position metrics for the diffusion model.

:class:`DiffusionMoleculeModule` trains :class:`DiffusionMoleculeModel` with an
epsilon-prediction DDPM objective: every step samples ``t ~ U(0, 1)`` and
``eps ~ N(0, I)``, noises the clean positions, wraps them into the cell, rebuilds
the PBC radius graph and minimizes ``MSE(eps_hat, eps)``. Validation runs the
same loss on a fixed grid of noise levels plus a cheap one-step denoising
``coord_rmse``.

The module also hosts the reverse sampler (:meth:`DiffusionMoleculeModule.sample`
/ :meth:`~DiffusionMoleculeModule.sample_many`) — DDPM or deterministic DDIM —
used for generation at the end of training and by the runner. The radius graph
is rebuilt from the current (noisy) coordinates at every reverse step and the
coordinates are kept inside the cell.

Position metrics (:func:`coord_rmse`, :func:`rdf_hist`, :func:`rdf_mad`,
:func:`min_pair_dist`) are provided here too; they are rotation/translation
robust enough for comparing generated conformations with ground-truth frames.
"""

from __future__ import annotations

import logging
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch_geometric.nn.aggr import MeanAggregation

from ..radius_graph import radius_graph_pbc, rebuild_pbc_edges
from .diffusion_model import DiffusionMoleculeModel
from .lightning_trainer import _json_safe

logger = logging.getLogger(__name__)

_mean_aggr = MeanAggregation()


# --- position metrics --------------------------------------------------------
def center_pos(pos: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
    """Subtract the (per-graph) centroid so RMSE is translation-invariant."""
    if batch is not None:
        return pos - _mean_aggr(pos, batch)[batch]
    return pos - pos.mean(dim=0, keepdim=True)


def coord_rmse(
    pred: torch.Tensor,
    truth: torch.Tensor,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Centered root-mean-square deviation (Å) between two coordinate sets.

    Centering removes the overall translation (per graph when ``batch`` is
    given). This is a conservative metric: MD frames of one molecule share a
    common orientation, so the raw (unrotated) RMSD is meaningful.
    """
    pred_c = center_pos(pred, batch)
    truth_c = center_pos(truth, batch)
    return torch.sqrt(F.mse_loss(pred_c, truth_c))


def min_image_pair_distances(pos: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """All-pairs minimum-image distances (Å), excluding self-pairs."""
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 3)
    box = box.to(pos.dtype)
    diff = diff - torch.round(diff / box) * box
    dist = diff.norm(dim=-1)  # (N, N)
    mask = torch.triu(torch.ones_like(dist), diagonal=1).bool()
    return dist[mask]


def pair_correlation(
    pos: torch.Tensor,
    box: torch.Tensor,
    dr: float = 0.1,
    rmax: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Determine the pair-correlation function ``g(r)`` of one periodic box.

    Every unordered pair of positions is assigned to a radial shell using its
    minimum-image distance.  The shell population is normalized by the
    population expected for an ideal gas with the same finite number of
    particles and cell volume, so a spatially uniform distribution has
    ``g(r)`` near one (away from sampling noise).

    ``rmax`` defaults to half the shortest cell length.  At or below this range
    each spherical shell is wholly representable under the minimum-image
    convention.  Returns ``(g_r, edges)``, where ``edges`` has one more entry
    than ``g_r`` and is expressed in Angstrom.
    """
    if pos.ndim != 2 or pos.shape[-1] != 3:
        raise ValueError(f"pos must have shape (N, 3), got {tuple(pos.shape)}")
    if dr <= 0:
        raise ValueError(f"dr must be positive, got {dr}")

    box = box.to(dtype=pos.dtype, device=pos.device).reshape(-1)
    if box.numel() != 3 or (box <= 0).any():
        raise ValueError("box must contain three positive orthorhombic lengths")
    if rmax is None:
        rmax = (box.min() / 2).item()
    if rmax <= 0:
        raise ValueError(f"rmax must be positive, got {rmax}")

    # Keep the final edge exactly at rmax: expanding the final shell beyond the
    # requested range would make its ideal-gas normalization inconsistent.
    nbins = max(int(rmax / dr), 1)
    edges = torch.linspace(
        0.0, rmax, nbins + 1, dtype=pos.dtype, device=pos.device
    )
    dist = min_image_pair_distances(pos, box)
    # `torch.histc` does not preserve the input device for all backends.  The
    # explicit bin assignment also makes the treatment of r == rmax clear:
    # it belongs in the final shell.
    bin_width = rmax / nbins
    bin_idx = torch.floor(dist / bin_width).to(torch.long).clamp(max=nbins - 1)
    valid = dist <= rmax
    counts = torch.bincount(bin_idx[valid], minlength=nbins).to(pos.dtype)

    shell_volumes = (4.0 * torch.pi / 3.0) * (edges[1:].pow(3) - edges[:-1].pow(3))
    n_particles = pos.shape[0]
    # There are N(N-1)/2 unordered pairs in a finite box.  This normalization
    # avoids the small-N bias of the common large-system rho*N/2 expression.
    expected = (
        n_particles * max(n_particles - 1, 0) / 2.0 * shell_volumes / box.prod()
    )
    return torch.where(expected > 0, counts / expected, torch.zeros_like(counts)), edges


def rdf_hist(
    pos: torch.Tensor,
    box: torch.Tensor,
    dr: float = 0.1,
    rmax: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible alias for :func:`pair_correlation`.

    Despite its historical name, the returned values are now the physically
    normalized radial distribution / pair-correlation function ``g(r)``.
    """
    return pair_correlation(pos, box, dr=dr, rmax=rmax)


def rdf_mad(hist_a: torch.Tensor, hist_b: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference between two pair-correlation functions."""
    return (hist_a - hist_b).abs().mean()


def min_pair_dist(pos: torch.Tensor) -> torch.Tensor:
    """Smallest inter-atomic distance (Å) in one structure (sanity check)."""
    if pos.shape[0] < 2:
        return torch.zeros((), dtype=pos.dtype, device=pos.device)
    d = torch.cdist(pos, pos)
    d = d + torch.eye(pos.shape[0], device=pos.device, dtype=pos.dtype) * 1e6
    return d.min()


# --- Lightning module --------------------------------------------------------
class DiffusionMoleculeModule(pl.LightningModule):
    """LightningModule for epsilon-prediction DDPM training of molecular positions.

    Handles the train/val/test loops, per-split loss + cheap denoising metric
    logging, the reverse sampler (DDPM / DDIM) and optimization (pluggable
    optimizer + LR scheduler, same pattern as
    :class:`SimpleLightningMoleculeModule`). All knobs are persisted in
    ``self.hparams`` except ``model`` (passed again to ``load_from_checkpoint``).
    The full resolved run config can be passed via ``config``.
    """

    def __init__(
        self,
        model: DiffusionMoleculeModel,
        radius: float,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
        scheduler_class: type | None = None,
        scheduler_kwargs: dict | None = None,
        scheduler_monitor: str = "val_loss",
        scheduler_interval: str = "epoch",
        val_t_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
        sample_steps: int = 100,
        sample_ddim: bool = False,
        sample_eta: float = 0.0,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.noise_schedule = model.noise_schedule  # cached for readability
        self.radius = radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_interval = scheduler_interval
        self.val_t_grid = tuple(val_t_grid)
        self.sample_steps = int(sample_steps)
        self.sample_ddim = bool(sample_ddim)
        self.sample_eta = float(sample_eta)
        config = _json_safe(config) if isinstance(config, dict) else None
        self.config = config
        self.save_hyperparameters(ignore=["model"])

    # -- diffusion plumbing ---------------------------------------------------
    @staticmethod
    def _batch_box(batch: Any) -> torch.Tensor:
        """Per-graph box lengths ``(B, 3)`` from a batch, robust to collation.

        ``H5MolecularDataset`` stores ``box`` as ``(1, 3)`` per graph, which PyG
        collates to ``(B, 3)``. If a ``(3,)`` box was stored instead, PyG
        collates it to ``(B * 3,)``; reshape it back to ``(B, 3)``.
        """
        box = getattr(batch, "box", None)
        if box is None:
            raise ValueError(
                "data.box is missing — the diffusion model needs per-graph "
                "PBC box lengths (set by H5MolecularDataset)"
            )
        if box.dim() == 1:
            return box.reshape(-1, 3)
        if box.dim() == 2 and box.shape[1] == 3:
            return box
        raise ValueError(
            f"unexpected batch.box shape {tuple(box.shape)}; expected (B, 3)"
        )

    def _corrupt(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Noise the clean positions and rebuild the PBC radius graph.

        Returns ``(x_noisy, edge_index, t, eps)``: in-cell noisy positions, the
        graph rebuilt from them, the per-graph timesteps and the raw Gaussian
        noise. The loss is computed on the unwrapped ``eps`` (standard; the
        boundary ambiguity after wrapping is negligible).
        """
        x0 = batch.pos  # (N, 3)
        B = batch.num_graphs
        device = x0.device
        t = torch.rand(B, device=device)  # (B,)
        eps = torch.randn_like(x0)  # (N, 3)
        sigma = self.noise_schedule.sigma(t)  # (B,)
        sigma_node = sigma[batch.batch].unsqueeze(-1)  # (N, 1)
        x_noisy = x0 + sigma_node * eps
        box = self._batch_box(batch)  # (B, 3)
        box_node = box[batch.batch]  # (N, 3)
        x_noisy = torch.remainder(x_noisy, box_node)  # wrap into the cell
        edge_index = rebuild_pbc_edges(x_noisy, batch.batch, box, self.radius)
        return x_noisy, edge_index, t, eps

    def _predict_eps(
        self, batch: Any, x_noisy: torch.Tensor, edge_index: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            x=batch.x,
            pos_noisy=x_noisy,
            edge_index=edge_index,
            batch=batch.batch,
            t=t,
            box=self._batch_box(batch),
        )

    # -- Lightning steps ------------------------------------------------------
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x_noisy, edge_index, t, eps = self._corrupt(batch)
        eps_hat = self._predict_eps(batch, x_noisy, edge_index, t)
        loss = F.mse_loss(eps_hat, eps)
        self.log(
            "train_loss", loss,
            on_step=True, on_epoch=True, prog_bar=True,
            batch_size=batch.num_graphs, sync_dist=True,
        )
        return loss

    def _eval_step(self, batch: Any, prefix: str) -> None:
        """Validation/test: loss on a fixed grid of noise levels + one-step RMSE."""
        losses, rmses = [], []
        box = self._batch_box(batch)
        for tv in self.val_t_grid:
            t = torch.full(
                (batch.num_graphs,), float(tv), device=batch.pos.device
            )
            eps = torch.randn_like(batch.pos)
            sigma = self.noise_schedule.sigma(t)
            sigma_node = sigma[batch.batch].unsqueeze(-1)
            x_noisy = batch.pos + sigma_node * eps
            box_node = box[batch.batch]
            x_noisy = torch.remainder(x_noisy, box_node)
            edge_index = rebuild_pbc_edges(x_noisy, batch.batch, box, self.radius)
            eps_hat = self._predict_eps(batch, x_noisy, edge_index, t)
            losses.append(F.mse_loss(eps_hat, eps))
            x0_hat = x_noisy - sigma_node * eps_hat  # one-step denoise estimate
            rmses.append(coord_rmse(x0_hat, batch.pos, batch.batch))
        self.log(
            f"{prefix}_loss", sum(losses) / len(losses),
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=batch.num_graphs, sync_dist=True,
        )
        self.log(
            f"{prefix}_coord_rmse", sum(rmses) / len(rmses),
            on_step=False, on_epoch=True, prog_bar=False,
            batch_size=batch.num_graphs, sync_dist=True,
        )

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._eval_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._eval_step(batch, "test")

    def on_train_epoch_end(self) -> None:
        """Log the current learning rate once per epoch (W&B / CSV)."""
        try:
            opt = self.optimizers()
            if isinstance(opt, (list, tuple)):
                opt = opt[0]
            lr = opt.param_groups[0]["lr"]
        except Exception:
            return
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    # Lightning's loosely-typed OptimizerLRScheduler return makes Pylance flag
    # the plain-dict scheduler form; suppress (same pattern as
    # SimpleLightningMoleculeModule, which is silent only by analysis luck).
    def configure_optimizers(self):  # type: ignore[override]
        optimizer_kwargs = dict(self.optimizer_kwargs or {})
        optimizer_kwargs.setdefault("lr", self.lr)
        optimizer_kwargs.setdefault("weight_decay", self.weight_decay)
        optimizer = self.optimizer_class(self.parameters(), **optimizer_kwargs)
        if self.scheduler_class is None:
            return optimizer
        scheduler = self.scheduler_class(optimizer, **(self.scheduler_kwargs or {}))
        lr_scheduler: dict = {
            "scheduler": scheduler,
            "interval": self.scheduler_interval,
        }
        if issubclass(self.scheduler_class, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler["monitor"] = self.scheduler_monitor
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    # -- sampling -------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        atom_types: torch.Tensor,
        cell: torch.Tensor,
        radius: float | None = None,
        steps: int | None = None,
        ddim: bool | None = None,
        eta: float | None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Generate one molecule conformation via reverse diffusion.

        Args:
            atom_types: Atomic numbers, shape ``(N,)``.
            cell: ``(3,)`` box lengths or ``(3, 3)`` lattice matrix.
            radius: Graph cutoff (defaults to the module's radius).
            steps: Number of reverse steps (defaults to ``sample_steps``).
            ddim: Use (deterministic) DDIM sampling (defaults to ``sample_ddim``).
            eta: DDIM stochasticity (only when ``ddim=True``; ``0`` = deterministic).
            seed: Optional RNG seed for reproducibility.
            device: Device to sample on (defaults to the module device).

        Returns:
            Generated positions of shape ``(N, 3)``, wrapped into the cell.
        """
        device = torch.device(device) if device is not None else self.device
        radius = self.radius if radius is None else radius
        steps = self.sample_steps if steps is None else int(steps)
        ddim = self.sample_ddim if ddim is None else bool(ddim)
        eta = self.sample_eta if eta is None else float(eta)

        atom_types = atom_types.to(device)
        cell = torch.as_tensor(cell, dtype=torch.float32, device=device)
        box = torch.diagonal(cell) if cell.ndim == 2 else cell
        N = atom_types.shape[0]
        schedule = self.noise_schedule

        # Dropout (and train-time batch-norm behavior) must be off while
        # sampling so the reverse process is deterministic given the seed.
        was_training = self.training
        self.eval()
        try:
            return self._sample_loop(
                atom_types, box, radius, steps, ddim, eta, N, schedule, seed
            )
        finally:
            if was_training:
                self.train()

    def _sample_loop(
        self,
        atom_types: torch.Tensor,
        box: torch.Tensor,
        radius: float,
        steps: int,
        ddim: bool,
        eta: float,
        N: int,
        schedule,
        seed: int | None,
    ) -> torch.Tensor:
        generator = (
            torch.Generator(device="cpu").manual_seed(seed)
            if seed is not None
            else None
        )
        x = torch.rand(N, 3, device=box.device, generator=generator) * box  # in-cell prior
        batch_vec = torch.zeros(N, dtype=torch.long, device=box.device)

        t_steps = torch.linspace(1.0, 0.0, steps + 1, device=box.device)
        for i in range(steps):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            x = torch.remainder(x, box)  # keep inside the cell
            edge_index = radius_graph_pbc(x, r=radius, lattice=box, loop=False)
            eps_hat = self._predict_eps_single(
                atom_types, x, edge_index, batch_vec, t_cur, box
            )  # (N, 3)

            ab_cur = schedule.alpha_bar(t_cur)
            ab_next = schedule.alpha_bar(t_next)
            sig_cur = schedule.sigma(t_cur)
            x0_hat = (x - sig_cur * eps_hat) / ab_cur.sqrt().clamp_min(1e-8)

            if ddim:
                if eta > 0:
                    sigma_t = eta * torch.sqrt(
                        ((1.0 - ab_next) / (1.0 - ab_cur).clamp_min(1e-8))
                        * (1.0 - ab_cur / ab_next.clamp_min(1e-8))
                    )
                else:
                    sigma_t = torch.zeros((), device=box.device)
                coef = torch.sqrt((1.0 - ab_next - sigma_t**2).clamp_min(0.0))
                x = ab_next.sqrt() * x0_hat + coef * eps_hat
                if eta > 0:
                    x = x + sigma_t * torch.randn_like(x)
            else:
                alpha_cur = (ab_cur / ab_next.clamp_min(1e-8)).clamp_max(1.0)
                beta = 1.0 - alpha_cur
                x = (x - beta / sig_cur.clamp_min(1e-8) * eps_hat) / alpha_cur.sqrt()
                post_std = torch.sqrt(
                    beta * (1.0 - ab_next) / (1.0 - ab_cur).clamp_min(1e-8)
                )
                x = x + post_std * torch.randn_like(x)
        return torch.remainder(x, box)

    def _predict_eps_single(
        self,
        atom_types: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_vec: torch.Tensor,
        t: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        """Model call for a single structure (batch-of-one view)."""
        return self.model(
            x=atom_types,
            pos_noisy=x,
            edge_index=edge_index,
            batch=batch_vec,
            t=t.unsqueeze(0),
            box=box.unsqueeze(0),
        )

    @torch.no_grad()
    def sample_many(
        self,
        atom_types: torch.Tensor,
        cell: torch.Tensor,
        n: int = 4,
        seed: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate ``n`` independent conformations; returns ``(n, N, 3)``."""
        if seed is None:
            seeds: list[int | None] = [None] * n
        else:
            seeds = [int(seed) + i for i in range(n)]
        return torch.stack(
            [self.sample(atom_types, cell, seed=s, **kwargs) for s in seeds]
        )
