"""PyTorch Lightning wrapper for training GNN scalar-regression models.

The wrapped ``model`` is expected to be a ``torch.nn.Module`` whose ``forward``
accepts the ``(x, edge_index, batch)`` triple from a PyG ``Batch`` and returns a
prediction of shape ``(num_graphs, 1)`` matching ``batch.y``.
"""

from collections.abc import Callable
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F


class SimpleLightningMoleculeModule(pl.LightningModule):
    """LightningModule for molecule-property (scalar regression) GNN training.

    Handles the train/val/test loop, per-split loss + metric logging, and
    optimization. The optimizer and LR scheduler are pluggable via
    ``optimizer_class``/``optimizer_kwargs`` and ``scheduler_class``/
    ``scheduler_kwargs`` (any ``torch.optim`` optimizer or
    ``torch.optim.lr_scheduler``, e.g. ``ReduceLROnPlateau``, ``StepLR``,
    ``CosineAnnealingLR``, ``OneCycleLR``). Any number of extra regression
    metrics can be logged alongside the loss via ``extra_metrics`` — a
    ``{name: callable}`` map, e.g. ``{"mae": F.l1_loss, "rmse": rmse}`` — each
    logged as ``{split}_{name}``. All knobs are persisted in ``self.hparams`` —
    except ``model``, ``loss_func`` and ``extra_metrics``, which are not
    trivially reconstructible from a checkpoint and must be passed again to
    ``load_from_checkpoint``.
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
            extra_metrics if extra_metrics is not None else {"mae": F.l1_loss}
        )
        self.save_hyperparameters(ignore=["model", "loss_func", "extra_metrics"])

    def forward(self, data: Any) -> torch.Tensor:
        """Predict graph-level properties from a batched PyG graph.

        Args:
            data: A ``torch_geometric.data.Batch`` (or ``Data``) providing ``x``
                (node features / atom types), ``edge_index`` (connectivity),
                ``batch`` (graph index per node) and, optionally, ``pos`` (node
                positions, used for geometric edge features).

        Returns:
            Predictions of shape ``(num_graphs, 1)``.
        """
        return self.model(
            data.x, data.edge_index, data.batch, getattr(data, "pos", None)
        )

    def _shared_step(self, batch: Any, prefix: str) -> torch.Tensor:
        y_hat = self(batch).view_as(batch.y)  # (B, 1) -> (B,), matching y
        y = batch.y
        loss = self.loss_func(y_hat, y)
        # Log the loss plus every configured extra metric (l1, rmse, ...) as
        # {prefix}_{name}. Any number of metrics can be added via extra_metrics.
        metrics = {f"{prefix}_loss": loss}
        for name, metric_fn in (self.extra_metrics or {}).items():
            metrics[f"{prefix}_{name}"] = metric_fn(y_hat, y)
        self.log_dict(
            metrics,
            on_step=prefix == "train",
            on_epoch=True,
            prog_bar=True,
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

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        return self(batch)

    def configure_optimizers(self):
        optimizer_kwargs = dict(self.optimizer_kwargs or {})
        # First-class defaults; explicit optimizer_kwargs take precedence.
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
        # Only ReduceLROnPlateau (and subclasses) requires a monitored metric.
        if issubclass(self.scheduler_class, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler["monitor"] = self.scheduler_monitor
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
