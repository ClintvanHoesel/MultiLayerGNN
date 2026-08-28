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

    When ``z_ordered=True`` the train/val/test corruption mirrors the
    point-by-point +z sampler (:meth:`sample_sequential_z`): the z-ordered box
    dataset marks the studied (target) molecule of every sample (with all
    molecules at/below its z as context), the context stays clean and only the
    target is noised at the graph's ``t``, with the loss computed on that
    target — so the model learns to denoise a molecule given the already-placed
    structure below it. No z-frontier has to be invented inside the trainer; the
    dataset provides it.
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
        z_ordered: bool = False,
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
        # Sequential (+z) training: the dataset marks the studied (target)
        # molecule of each sample; only that target is noised (see
        # ``_corrupt_z_ordered``), matching ``sample_sequential_z``.
        self.z_ordered = bool(z_ordered)
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

    def _corrupt_z_ordered(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequential z-ordered corruption — context clean, targets noisy.

        Reads the per-node ``target_mask`` supplied by the z-ordered box dataset
        (the studied molecule of each sample, with every molecule at/below its z
        kept as clean context): only the marked target nodes are noised at the
        graph's ``t``, the context stays clean, and the model is asked to predict
        ``eps`` for the targets only (``mask``). The graph is rebuilt from the
        noisy coordinates. Returns ``(x_noisy, edge_index, t, eps, mask)``.
        """
        x0 = batch.pos  # (N, 3)
        B = batch.num_graphs
        device = x0.device
        mask = getattr(batch, "target_mask", None)
        if mask is None:
            raise ValueError(
                "z-ordered corruption requires a per-node target_mask (set by "
                "ZOrderedBoxMolecularDataset)"
            )
        mask = mask.to(device).bool()
        t = torch.rand(B, device=device)  # per-graph target noise level
        eps = torch.randn_like(x0) * mask.to(x0.dtype).unsqueeze(-1)  # targets only
        sigma = self.noise_schedule.sigma(t)  # (B,)
        sigma_node = sigma[batch.batch].unsqueeze(-1)  # (N, 1)
        x_noisy = x0 + sigma_node * eps  # context: sigma * 0 -> clean
        box = self._batch_box(batch)  # (B, 3)
        box_node = box[batch.batch]  # (N, 3)
        x_noisy = torch.remainder(x_noisy, box_node)  # wrap into the cell
        edge_index = rebuild_pbc_edges(x_noisy, batch.batch, box, self.radius)
        return x_noisy, edge_index, t, eps, mask

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
        if self.z_ordered:
            # Sequential (+z) corruption: the z-ordered box dataset marks the
            # studied (target) molecule per sample; it is noised at the graph's
            # t while the molecules below it (context) stay clean — matching
            # sample_sequential_z. The loss is over the target nodes only.
            x_noisy, edge_index, t, eps, mask = self._corrupt_z_ordered(batch)
        else:
            x_noisy, edge_index, t, eps = self._corrupt(batch)
            mask = None
        eps_hat = self._predict_eps(batch, x_noisy, edge_index, t)
        loss = (
            F.mse_loss(eps_hat[mask], eps[mask])
            if mask is not None
            else F.mse_loss(eps_hat, eps)
        )
        self.log(
            "train_loss", loss,
            on_step=True, on_epoch=True, prog_bar=True,
            batch_size=batch.num_graphs, sync_dist=True,
        )
        return loss

    def _eval_step(self, batch: Any, prefix: str) -> None:
        """Validation/test: loss on a fixed grid of noise levels + one-step RMSE.

        In z-ordered mode the batch carries the dataset's per-node
        ``target_mask`` (the studied molecule per sample): it is used for every
        noise level of the grid — the lower context molecules stay clean, the
        target is noised and the loss/RMSE is computed on that target only.
        """
        losses, rmses = [], []
        box = self._batch_box(batch)
        mask = (
            getattr(batch, "target_mask", None)
            if self.z_ordered
            else None
        )
        for tv in self.val_t_grid:
            t = torch.full(
                (batch.num_graphs,), float(tv), device=batch.pos.device
            )
            eps = torch.randn_like(batch.pos)
            if mask is not None:
                eps = eps * mask.to(eps.dtype).unsqueeze(-1)
            sigma = self.noise_schedule.sigma(t)
            sigma_node = sigma[batch.batch].unsqueeze(-1)
            x_noisy = batch.pos + sigma_node * eps
            box_node = box[batch.batch]
            x_noisy = torch.remainder(x_noisy, box_node)
            edge_index = rebuild_pbc_edges(x_noisy, batch.batch, box, self.radius)
            eps_hat = self._predict_eps(batch, x_noisy, edge_index, t)
            x0_hat = x_noisy - sigma_node * eps_hat  # one-step denoise estimate
            if mask is not None:
                losses.append(F.mse_loss(eps_hat[mask], eps[mask]))
                rmses.append(
                    coord_rmse(x0_hat[mask], batch.pos[mask], batch.batch[mask])
                )
            else:
                losses.append(F.mse_loss(eps_hat, eps))
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
            torch.Generator(device=box.device).manual_seed(seed)
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
            x = self._reverse_update(
                x, eps_hat, t_cur, t_next, schedule, ddim, eta, generator
            )
        return torch.remainder(x, box)

    @staticmethod
    def _reverse_update(
        x: torch.Tensor,
        eps_hat: torch.Tensor,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
        schedule,
        ddim: bool,
        eta: float,
        generator=None,
    ) -> torch.Tensor:
        """One reverse-diffusion update (DDPM or DDIM) for a (sub)set of nodes.

        The update is pure per-node math — every node's new value depends only
        on its own ``x`` / predicted ``eps`` and the (scalar) schedule
        quantities — so it can be applied to a full batch of nodes
        (:meth:`_sample_loop`) or to a single newly generated node
        (:meth:`_sequential_z_step`). ``generator`` is optional: when given, all
        posterior Gaussian draws use it (fully seeded, reproducible sampling);
        otherwise the global RNG is used.
        """
        ab_cur = schedule.alpha_bar(t_cur)
        ab_next = schedule.alpha_bar(t_next)
        sig_cur = schedule.sigma(t_cur)
        x0_hat = (x - sig_cur * eps_hat) / ab_cur.sqrt().clamp_min(1e-8)

        def _noise_like(like: torch.Tensor) -> torch.Tensor:
            if generator is not None:
                return torch.randn_like(like, generator=generator)
            return torch.randn_like(like)

        if ddim:
            if eta > 0:
                sigma_t = eta * torch.sqrt(
                    ((1.0 - ab_next) / (1.0 - ab_cur).clamp_min(1e-8))
                    * (1.0 - ab_cur / ab_next.clamp_min(1e-8))
                )
            else:
                sigma_t = torch.zeros((), device=x.device)
            coef = torch.sqrt((1.0 - ab_next - sigma_t**2).clamp_min(0.0))
            x_next = ab_next.sqrt() * x0_hat + coef * eps_hat
            if eta > 0:
                x_next = x_next + sigma_t * _noise_like(x_next)
        else:
            alpha_cur = (ab_cur / ab_next.clamp_min(1e-8)).clamp_max(1.0)
            beta = 1.0 - alpha_cur
            x_next = (x - beta / sig_cur.clamp_min(1e-8) * eps_hat) / alpha_cur.sqrt()
            post_std = torch.sqrt(
                beta * (1.0 - ab_next) / (1.0 - ab_cur).clamp_min(1e-8)
            )
            x_next = x_next + post_std * _noise_like(x_next)
        return x_next

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

    # -- sequential (point-by-point in +z) sampling --------------------------
    def _z_frontier(
        self, pos_ctx: torch.Tensor, bottom_z_exclusion: float, z_step: float
    ) -> float:
        """Current z-frontier the next generated molecule must stay above.

        ``max(bottom_z_exclusion, max z of the structure so far) + z_step`` —
        for an empty structure this reduces to ``bottom_z_exclusion + z_step``,
        so the first generated molecule is kept above the excluded bottom layer.
        """
        z_max = float(pos_ctx[:, 2].max()) if pos_ctx.numel() else float("-inf")
        return max(bottom_z_exclusion, z_max) + z_step

    def _sequential_z_step(
        self,
        species_new: torch.Tensor,
        type_ctx: torch.Tensor,
        pos_ctx: torch.Tensor,
        box: torch.Tensor,
        radius: float,
        steps: int,
        ddim: bool,
        eta: float,
        schedule,
        t_steps: torch.Tensor,
        z_frontier: float,
        generator,
    ) -> torch.Tensor:
        """Reverse-diffuse ONE new molecule with the structure-so-far frozen.

        The new node is initialized inside the cell with its z uniform in
        ``[z_frontier, box_z]`` (the ordering constraint is already respected by
        the prior), then every reverse step denoises ONLY the new node while the
        previously generated molecules (``pos_ctx`` / ``type_ctx``) stay fixed
        in the graph as conditioning context — exactly the input the model saw
        during z-ordered training (context at clean positions, one noisy node,
        all nodes sharing the graph's ``t``). After each step the new node is
        wrapped back into the cell and its z clamped to ``>= z_frontier``, so
        the constraint holds throughout sampling rather than only as
        post-processing.
        """
        k = pos_ctx.shape[0]
        box_z = float(box[2])
        if z_frontier > box_z:
            logger.warning(
                "sequential_z: z-frontier %.3f exceeds the cell height %.3f — "
                "no room above the current structure; the next molecule is "
                "pinned to the top of the cell",
                z_frontier,
                box_z,
            )
        lo = min(z_frontier, box_z)

        # In-cell prior respecting the ordering constraint at initialization.
        x_new = torch.empty(1, 3, device=box.device)
        x_new[0, 0] = box[0] * torch.rand(1, device=box.device, generator=generator)
        x_new[0, 1] = box[1] * torch.rand(1, device=box.device, generator=generator)
        x_new[0, 2] = lo + (box_z - lo) * torch.rand(
            1, device=box.device, generator=generator
        )

        types_all = torch.cat([type_ctx, species_new])  # (k + 1,)
        batch_vec = torch.zeros(k + 1, dtype=torch.long, device=box.device)
        box_graph = box.unsqueeze(0)  # (1, 3) per-graph box

        for i in range(steps):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            x_all = torch.cat([pos_ctx, x_new])  # (k + 1, 3), in-cell
            edge_index = radius_graph_pbc(x_all, r=radius, lattice=box, loop=False)
            eps_hat = self.model(
                x=types_all,
                pos_noisy=x_all,
                edge_index=edge_index,
                batch=batch_vec,
                t=t_cur.unsqueeze(0),
                box=box_graph,
            )  # (k + 1, 3)
            eps_new = eps_hat[-1:]  # (1, 3) — denoise only the new node
            x_new = self._reverse_update(
                x_new, eps_new, t_cur, t_next, schedule, ddim, eta, generator
            )
            # Hard z-ordering / in-cell constraint during sampling.
            x_new = torch.remainder(x_new, box)
            x_new[0, 2] = x_new[0, 2].clamp(lo, box_z)
        return x_new  # (1, 3), in-cell, z in [z_frontier, box_z]

    @torch.no_grad()
    def sample_sequential_z(
        self,
        atom_types: torch.Tensor,
        cell: torch.Tensor,
        num_points: int | None = None,
        bottom_z_exclusion: float = 0.0,
        z_step: float = 0.0,
        initial_pos: torch.Tensor | None = None,
        initial_types: torch.Tensor | None = None,
        radius: float | None = None,
        steps: int | None = None,
        ddim: bool | None = None,
        eta: float | None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Generate molecule positions point-by-point, each above the previous in +z.

        Unlike :meth:`sample` — which reverse-diffuses every node of a fixed
        structure simultaneously — this generator places the molecules one at a
        time along +z:

        * an empty structure is (optionally) seeded with ``initial_pos`` /
          ``initial_types`` (fixed context, never updated);
        * the current z-frontier is ``max(bottom_z_exclusion, max z so far)``
          plus ``z_step`` separation when given;
        * the next molecule is reverse-diffused with every previously generated
          molecule kept fixed in the graph as context, its COM z-coordinate
          constrained to stay at/above the frontier at every sampling step;
        * the molecule is appended and the frontier advances.

        The constraint is enforced during diffusion sampling (initialization in
        ``[z_frontier, box_z]`` plus a per-step z clamp), never as a
        post-processing re-sort. Positions are generated inside the cell; in the
        molecules-as-nodes (box) representation the ordering therefore applies
        to the molecule center-of-mass z-coordinates, i.e. the molecular
        representation already used by the model.

        Args:
            atom_types: Species of the molecules to generate, shape ``(N,)``.
                Only the first ``num_points`` entries are used.
            cell: ``(3,)`` box lengths or ``(3, 3)`` lattice matrix.
            num_points: Number of molecules to generate (defaults to ``N``).
            bottom_z_exclusion: Minimum z (Angstrom) below which no molecule is
                generated — the excluded bottom layer. The first generated
                molecule satisfies ``z >= bottom_z_exclusion``.
            z_step: Optional minimum separation along z between consecutive
                molecules (``0`` = only require each new molecule at/above the
                current z-frontier; the model freely chooses the spacing).
            initial_pos: Optional fixed initial structure, shape ``(M, 3)``,
                used as frozen context (never updated). The first generated
                molecule must stay above ``max(bottom_z_exclusion, max z of the
                initial structure)`` (+ ``z_step``).
            initial_types: Species of the initial structure, shape ``(M,)``
                (required when ``initial_pos`` is given).
            radius / steps / ddim / eta / seed / device: as :meth:`sample`.

        Returns:
            Generated molecule positions of shape ``(num_points, 3)``, ordered
            by generation step along +z. Each satisfies
            ``pos[i, 2] >= max(bottom_z_exclusion, max z of everything placed
            before it) (+ z_step)``.
        """
        device = torch.device(device) if device is not None else self.device
        radius = self.radius if radius is None else radius
        steps = self.sample_steps if steps is None else int(steps)
        ddim = self.sample_ddim if ddim is None else bool(ddim)
        eta = self.sample_eta if eta is None else float(eta)
        bottom_z_exclusion = float(bottom_z_exclusion)
        z_step = float(z_step)

        atom_types = atom_types.to(device)
        cell = torch.as_tensor(cell, dtype=torch.float32, device=device)
        box = torch.diagonal(cell) if cell.ndim == 2 else cell
        if box.numel() != 3 or (box <= 0).any():
            raise ValueError("cell must give three positive orthorhombic box lengths")
        N = atom_types.shape[0]
        num_points = N if num_points is None else int(num_points)
        if num_points < 0:
            raise ValueError("num_points must be >= 0")
        if num_points > N:
            raise ValueError(
                f"num_points ({num_points}) exceeds the number of provided "
                f"species ({N})"
            )

        if initial_pos is not None:
            pos_ctx = torch.as_tensor(initial_pos, dtype=torch.float32, device=device)
            if pos_ctx.ndim != 2 or pos_ctx.shape[1] != 3:
                raise ValueError("initial_pos must have shape (M, 3)")
            if initial_types is None:
                raise ValueError("initial_types is required when initial_pos is given")
            type_ctx = torch.as_tensor(
                initial_types, dtype=atom_types.dtype, device=device
            ).reshape(-1)
            if type_ctx.shape[0] != pos_ctx.shape[0]:
                raise ValueError(
                    "initial_types and initial_pos must have the same number of nodes"
                )
        else:
            pos_ctx = torch.empty(0, 3, device=device)
            type_ctx = torch.empty(0, dtype=atom_types.dtype, device=device)
        pos_ctx = torch.remainder(pos_ctx, box)  # keep the context in the cell

        schedule = self.noise_schedule
        # Dropout (and train-time batch-norm behavior) must be off while
        # sampling so the reverse process is deterministic given the seed.
        was_training = self.training
        self.eval()
        try:
            generator = (
                torch.Generator(device=box.device).manual_seed(seed)
                if seed is not None
                else None
            )
            t_steps = torch.linspace(1.0, 0.0, steps + 1, device=box.device)
            generated: list[torch.Tensor] = []
            for k in range(num_points):
                z_frontier = self._z_frontier(pos_ctx, bottom_z_exclusion, z_step)
                x_new = self._sequential_z_step(
                    atom_types[k : k + 1],
                    type_ctx,
                    pos_ctx,
                    box,
                    radius,
                    steps,
                    ddim,
                    eta,
                    schedule,
                    t_steps,
                    z_frontier,
                    generator,
                )  # (1, 3)
                pos_ctx = torch.cat([pos_ctx, x_new])
                type_ctx = torch.cat([type_ctx, atom_types[k : k + 1]])
                generated.append(x_new)
            if generated:
                return torch.cat(generated, dim=0)  # (num_points, 3)
            return torch.empty(0, 3, device=box.device)
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def sample_sequential_z_many(
        self,
        atom_types: torch.Tensor,
        cell: torch.Tensor,
        n: int = 4,
        seed: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate ``n`` independent point-by-point (z-ordered) structures.

        Returns ``(n, num_points, 3)`` where every structure is produced by
        :meth:`sample_sequential_z` with its own seed (``seed + i``) — i.e.
        multiple independent samples, each advancing its own z-frontier,
        mirroring :meth:`sample_many`. All extra kwargs (``num_points``,
        ``bottom_z_exclusion``, ``z_step``, ``initial_pos``, ...) are forwarded.
        """
        if seed is None:
            seeds: list[int | None] = [None] * n
        else:
            seeds = [int(seed) + i for i in range(n)]
        return torch.stack(
            [
                self.sample_sequential_z(atom_types, cell, seed=s, **kwargs)
                for s in seeds
            ]
        )
