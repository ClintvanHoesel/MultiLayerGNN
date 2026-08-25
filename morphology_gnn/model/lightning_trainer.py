"""PyTorch Lightning wrapper for training GNN scalar-regression models.

The wrapped ``model`` is expected to be a ``torch.nn.Module`` whose ``forward``
accepts the ``(x, edge_index, batch)`` triple from a PyG ``Batch`` and returns a
prediction of shape ``(num_graphs, 1)`` matching ``batch.y``.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def r2_score(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Coefficient of determination (R²) between predictions and targets.

    ``R² = 1 - SS_res / SS_tot``. This is the per-call (e.g. per-batch) value;
    during training Lightning averages it over batches, so the authoritative
    number is the one computed over the full dataset (see ``run_training.main``).

    Follows the scikit-learn convention for degenerate (constant) targets:
    returns 1.0 when the predictions are exact and 0.0 otherwise, avoiding NaN.
    """
    y_hat = y_hat.reshape_as(y)
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    # Avoid .item() (forces a GPU->CPU sync); handle degenerate (constant)
    # targets with pure tensor ops: return 1.0 when predictions are exact and
    # 0.0 otherwise. torch.where discards the unselected branch, so the NaN
    # from 0/0 in the fallback never leaks into the result.
    return torch.where(
        ss_tot == 0,
        torch.where(
            ss_res == 0,
            torch.ones_like(ss_res),
            torch.zeros_like(ss_res),
        ),
        1.0 - ss_res / ss_tot,
    )


def _json_safe(value: Any) -> Any:
    """Best-effort JSON round-trip so config hparams are W&B/checkpoint-safe.

    The resolved config may contain values (e.g. numpy scalars, tuples) that
    ``wandb.config`` / Lightning hparams dislike; ``default=str`` guarantees a
    serializable copy. Falls back to ``None`` on failure.
    """
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return None


class SimpleLightningMoleculeModule(pl.LightningModule):
    """LightningModule for molecule-property (scalar regression) GNN training.

    Handles the train/val/test loop, per-split loss + metric logging, and
    optimization. The optimizer and LR scheduler are pluggable via
    ``optimizer_class``/``optimizer_kwargs`` and ``scheduler_class``/
    ``scheduler_kwargs`` (any ``torch.optim`` optimizer or
    ``torch.optim.lr_scheduler``, e.g. ``ReduceLROnPlateau``, ``StepLR``,
    ``CosineAnnealingLR``, ``OneCycleLR``). Any number of extra regression
    metrics can be logged alongside the loss via ``extra_metrics`` — a
    ``{name: callable}`` map, e.g. ``{"mae": F.l1_loss, "r2": r2_score}`` — each
    logged as ``{split}_{name}``. If ``target_tags`` (one W&B-friendly label per
    target property, e.g. ``["positive_vip", "homo"]``) is provided, the loss
    and every extra metric are also computed per target and logged as
    ``{split}_target_{tag}_{name}``. All knobs are persisted in ``self.hparams`` —
    except ``model``, ``loss_func`` and ``extra_metrics``, which are not
    trivially reconstructible from a checkpoint and must be passed again to
    ``load_from_checkpoint``. The full resolved run config can be passed via
    ``config``; it is stored in ``self.hparams["config"]`` so it persists in the
    checkpoint and is auto-logged to W&B (Overview -> Config) by the
    ``WandbLogger`` at fit time.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
        scheduler_class: type | None = None,
        scheduler_kwargs: dict | None = None,
        scheduler_monitor: str = "val_loss",
        scheduler_interval: str = "epoch",
        extra_metrics: (
            dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | None
        ) = None,
        target_tags: list[str] | None = None,
        target_mean: torch.Tensor | list | None = None,
        target_std: torch.Tensor | list | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_func = loss_func
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_interval = scheduler_interval
        # Default to MAE (keeps the previous behaviour); pass {} to log only loss.
        self.extra_metrics = (
            extra_metrics
            if extra_metrics is not None
            else {"mae": F.l1_loss, "r2": r2_score}
        )
        # One W&B-friendly label per target property (e.g. ["positive_vip"]).
        # When set, the loss and every extra metric are also logged per target.
        self.target_tags = list(target_tags) if target_tags is not None else None
        # Target standardization: per-target mean/std (fit on the training set).
        # The loss is computed on standardized targets (well-conditioned for
        # near-constant / large-magnitude targets) and predictions are
        # un-standardized via :meth:`denormalize_targets`. Defaults to identity.
        num_targets = getattr(getattr(model, "lin", None), "out_features", 1)
        self.register_buffer(
            "target_mean",
            (
                torch.zeros(num_targets)
                if target_mean is None
                else torch.as_tensor(target_mean, dtype=torch.float32)
            ),
        )
        self.register_buffer(
            "target_std",
            (
                torch.ones(num_targets)
                if target_std is None
                else torch.as_tensor(target_std, dtype=torch.float32)
            ),
        )
        # Full resolved config (data/model/training/logging sections). Kept in
        # hparams so it (a) persists in the checkpoint, (b) is auto-logged to
        # W&B by the WandbLogger at fit time (Overview -> Config), and (c) is
        # available as self.hparams["config"]. JSON-normalized for safety.
        config = _json_safe(config) if isinstance(config, dict) else None
        self.config = config
        self.save_hyperparameters(
            ignore=[
                "model",
                "loss_func",
                "extra_metrics",
                "target_mean",
                "target_std",
            ]
        )

    def forward(self, data: Any) -> torch.Tensor:
        """Predict graph-level properties from a batched PyG graph.

        Args:
            data: A ``torch_geometric.data.Batch`` (or ``Data``) providing ``x``
                (node features / atom types), ``edge_index`` (connectivity),
                ``batch`` (graph index per node) and, optionally, ``pos`` (node
                positions, used for geometric edge features). In context mode it
                also carries ``mol_index`` / ``mol_is_query`` (per-molecule
                readout) and ``box`` (PBC minimum-image edge features), which are
                forwarded to the model when present.

        Returns:
            Predictions of shape ``(num_graphs, 1)``.
        """
        kw: dict = {}
        for attr in ("mol_index", "mol_is_query"):
            if hasattr(data, attr):
                kw[attr] = getattr(data, attr)
        if hasattr(data, "box"):
            kw["box"] = data.box
        return self.model(
            data.x, data.edge_index, data.batch, getattr(data, "pos", None), **kw
        )

    def normalize_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Standardize targets to zero-mean / unit-variance per target column.

        Accepts the flattened ``y`` (``(B*T,)`` / ``(B,)`` as produced by PyG
        batching) and returns it flattened and standardized.
        """
        y = y.reshape(-1, self.target_mean.numel())
        return ((y - self.target_mean) / self.target_std).reshape(-1)

    def denormalize_targets(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Map standardized predictions back to the original target scale.

        Args:
            y_hat: Model output of shape ``(num_graphs, num_targets)``.

        Returns:
            Predictions of shape ``(num_graphs, num_targets)`` in original units.
        """
        return (
            y_hat.reshape(-1, self.target_mean.numel()) * self.target_std
            + self.target_mean
        )

    def _shared_step(self, batch: Any, prefix: str) -> torch.Tensor:
        y_hat = self(batch)  # (num_graphs, num_targets), in standardized space
        y = batch.y  # (num_graphs,) or (num_graphs*num_targets,) after PyG flattening
        # Align shapes for the aggregate loss: PyG flattens a per-graph `y` of
        # shape (T,) into (B*T,) when batching, so reshape the model output to
        # match whatever `y` looks like.
        y_hat_flat = y_hat.view_as(y)
        # The loss is computed on standardized targets (well-conditioned for
        # near-constant / large-magnitude targets); predictions and the extra
        # metrics are reported in the original (physical) target units.
        y_norm = self.normalize_targets(y)
        loss = self.loss_func(y_hat_flat, y_norm)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[%s] batch graphs=%d loss=%.6f",
                prefix,
                batch.num_graphs,
                float(loss),
            )
        # Aggregate metrics: loss plus every configured extra metric (l1, rmse,
        # ...) as {prefix}_{name}, computed on denormalized predictions so the
        # logged values are in physical target units.
        n_targets = self.target_mean.numel()
        y_hat_raw = self.denormalize_targets(y_hat)  # (B, T) original units
        y_raw = y.reshape(-1, n_targets)  # (B, T) original units
        metrics = {f"{prefix}_loss": loss}
        for name, metric_fn in (self.extra_metrics or {}).items():
            metrics[f"{prefix}_{name}"] = metric_fn(y_hat_raw, y_raw)
        self.log_dict(
            metrics,
            on_step=prefix == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
            sync_dist=True,
        )

        # Per-target metrics ({prefix}_target_{tag}_{name}): split both
        # predictions and targets into one column per property. Handles the
        # single-target case (target_tags of length 1, y of shape (B,)) and the
        # flattened multi-target case (PyG packs y as (B*T,) -> (B, T)).
        if self.target_tags:
            y_norm_m = y_norm.reshape(-1, n_targets)  # standardized (B, T)
            per_target = {}
            for t, tag in enumerate(self.target_tags):
                per_target[f"{prefix}_target_{tag}_loss"] = self.loss_func(
                    y_hat[:, t], y_norm_m[:, t]
                )
                for name, metric_fn in (self.extra_metrics or {}).items():
                    per_target[f"{prefix}_target_{tag}_{name}"] = metric_fn(
                        y_hat_raw[:, t], y_raw[:, t]
                    )
            self.log_dict(
                per_target,
                on_step=prefix == "train",
                on_epoch=True,
                prog_bar=False,
                batch_size=batch.num_graphs,
                sync_dist=True,
            )
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

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

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        # Return predictions in the original target units.
        return self.denormalize_targets(self(batch))

    def configure_optimizers(self):
        optimizer_kwargs = dict(self.optimizer_kwargs or {})
        # First-class defaults; explicit optimizer_kwargs take precedence.
        optimizer_kwargs.setdefault("lr", self.lr)
        optimizer_kwargs.setdefault("weight_decay", self.weight_decay)
        optimizer = self.optimizer_class(self.parameters(), **optimizer_kwargs)
        logger.debug(
            "configure_optimizers: optimizer=%s scheduler=%s",
            type(optimizer).__name__,
            self.scheduler_class.__name__ if self.scheduler_class is not None else None,
        )

        if self.scheduler_class is None:
            return optimizer

        scheduler = self.scheduler_class(optimizer, **(self.scheduler_kwargs or {}))
        lr_scheduler: dict = {
            "scheduler": scheduler,
            "interval": self.scheduler_interval,
        }
        # Only ReduceLROnPlateau (and subclasses) requires a monitored metric.
        if issubclass(self.scheduler_class, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler["monitor"] = self.scheduler_monitor
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
