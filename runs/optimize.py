"""Hyperparameter optimization for the scalar GNN trainer (``run_training.py``).

Drives :mod:`run_training`'s building blocks with an Optuna study. Every trial:

1. samples a hyperparameter config on top of a base config file,
2. builds a ``ScalarMoleculeModel`` + ``SimpleLightningMoleculeModule``,
3. trains for a capped number of epochs with early stopping (and optional
   Optuna pruning), and
4. returns a validation objective (default ``val_mae``; ``val_r2`` maximizes).

The best config is written to ``<outdir>/hpo/best_config.yaml``, ready to feed
straight back into ``run_training.py --config``.

Usage (must use the ``torch`` conda environment — nothing else is touched)::

    conda run -n torch python runs/optimize.py \\
        --config runs/config.yaml --n-trials 20 --objective val_mae

The search space lives in ``_DEFAULT_SEARCH_SPACE`` (below) and can be overridden
with ``--search-space some.yaml`` (see ``runs/search_space.yaml`` for the
format). Every dotted key is a config path exactly as in ``run_training.py``
deep overrides (e.g. ``model.hidden_dim``, ``training.lr``).

Optional: pass ``--wandb-project <name>`` to log every trial (with
human-friendly names) plus a study summary to W&B.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from datetime import datetime

# Make the project root importable regardless of how this script is launched.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Configure the package logger before importing morphology_gnn (via run_training)
# so import-time warnings (e.g. CUDA fallback) are captured. Level comes from
# MGN_LOG_LEVEL, else WARNING; upgraded to INFO (or --log-level) in main().
from morphology_gnn._logging import configure_logging  # noqa: E402

configure_logging(level=os.environ.get("MGN_LOG_LEVEL"))

# Stdlib logger for the CLI (named `log` to avoid clashing with Lightning loggers).
log = logging.getLogger("morphology_gnn.runs.optimize")

import lightning.pytorch as pl
import optuna
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from run_training import (  # noqa: E402  (sys.path fix above)
    DEFAULT_CONFIG,
    _ensure_wandb_auth,
    _finalize_wandb,
    _log_config_to_wandb,
    _log_yaml_files_to_wandb,
    _make_run_name,
    build_dataset,
    build_loaders,
    build_model,
    build_module,
    compute_metrics,
    configure_cuda,
    deep_merge,
    fit_target_scaler,
    gradient_clip_kwargs,
    load_config,
    normalize_targets,
    require_radius,
    sanitize_name,
    set_nested,
    set_seed,
)

# --- default search space -----------------------------------------------------
# Mapping of dotted config path -> Optuna sampling spec. Supported types:
#   categorical: {"type": "categorical", "choices": [...]}
#   int:         {"type": "int", "low": ..., "high": ..., "step": ...}
#   float:       {"type": "float", "low": ..., "high": ..., "step": ...}
#   log_float:   {"type": "log_float", "low": ..., "high": ...}
#   log_int:     {"type": "log_int", "low": ..., "high": ...}
_DEFAULT_SEARCH_SPACE = {
    # "model.hidden_dim": {"type": "categorical", "choices": [64, 128, 256]},
    "model.num_layers": {"type": "int", "low": 1, "high": 4},
    # "model.heads": {"type": "categorical", "choices": [1, 2, 4, 8]},
    # "model.num_rbf": {"type": "categorical", "choices": [25, 50, 75, 100]},
    # "model.rbf_kwargs.rbf_class": {
    #     "type": "categorical",
    #     "choices": ["GaussianRBF", "ExpNormalRBF", "BesselRBF", "ChebychevRBF"],
    # },
    # "model.rbf_kwargs.cutoff_fn": {
    #     "type": "categorical",
    #     "choices": ["CosineEnvelope", "PolynomialEnvelope"],
    # },
    # "model.dropout": {"type": "float", "low": 0.0, "high": 0.5},
    # "model.act": {
    #     "type": "categorical",
    #     "choices": ["gelu", "mish", "relu", "silu"],
    # },
    # "model.conv_class": {
    #     "type": "categorical",
    #     "choices": ["GATConv", "GCNConv", "SAGEConv"],
    # },
    # "model.use_edge_features": {
    #     "type": "categorical",
    #     "choices": [False, True],
    # },
    # "training.lr": {"type": "log_float", "low": 1e-5, "high": 1e-2},
    # "training.batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    # "training.weight_decay": {"type": "log_float", "low": 1e-6, "high": 1e-2},
    # "training.optimizer_class": {
    #     "type": "categorical",
    #     "choices": ["Adam", "AdamW"],
    # },
}

# val_mae / val_loss are lower-better; val_r2 is higher-better (the script
# auto-sets the study direction to maximize when val_r2 is selected).
OBJECTIVE_CHOICES = ("val_mae", "val_loss", "val_r2")

# Cache datasets per (files, target, radius) so HPO trials don't re-parse HDF5.
_DATASET_CACHE: dict = {}


# --- helpers ------------------------------------------------------------------
def _load_search_space(path: str | None) -> dict:
    if not path:
        return copy.deepcopy(_DEFAULT_SEARCH_SPACE)
    import yaml

    with open(path) as f:
        space = yaml.safe_load(f) or {}
    return space


def _suggest(space: dict, trial: optuna.Trial) -> dict:
    """Sample one config-override dict from the search space for this trial."""
    overrides = {}
    for path, spec in space.items():
        name = path.replace(".", "__")
        kind = spec["type"]
        if kind == "categorical":
            value = trial.suggest_categorical(name, spec["choices"])
        elif kind == "int":
            value = trial.suggest_int(
                name, spec["low"], spec["high"], step=spec.get("step", 1)
            )
        elif kind == "log_int":
            value = trial.suggest_int(name, spec["low"], spec["high"], log=True)
        elif kind == "float":
            value = trial.suggest_float(
                name, spec["low"], spec["high"], step=spec.get("step")
            )
        elif kind == "log_float":
            value = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        else:
            raise ValueError(f"unknown search-space type: {kind!r}")
        overrides[path] = value
    return overrides


def _get_dataset(config: dict):
    """Build (and cache) the scalar-regression dataset for one trial.

    Cache key includes the dataset layout so molecular and box builds of
    the same files/targets/radius do not collide.
    """
    files = config["data"]
    if isinstance(files, str):
        files = [files]
    targets = normalize_targets(config["target"])
    key = (
        config.get("dataset", "molecular"),
        tuple(files),
        tuple(targets),
        float(config["radius"]),
    )
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = build_dataset(config)
    return _DATASET_CACHE[key]


def _evaluate(module, loader, targets=None) -> dict:
    """Aggregate + per-target metrics over a loader (deterministic)."""
    module.eval()
    device = next(module.parameters()).device
    ys, preds, groups = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y_hat = module(batch)  # (num_graphs, num_targets), standardized
            # PyG flattens per-graph `y` (T,) into (B*T,) when batching; reshape
            # back to (num_graphs, num_targets) so it aligns with the model head.
            ys.append(batch.y.view_as(y_hat))
            # Un-standardize predictions into the original target units.
            preds.append(module.denormalize_targets(y_hat))
            # Per-graph group labels (material/species or molecule id) for
            # group-aware R2 (not inflated by pooling different materials).
            for attr in ("species_name", "mol_name"):
                if hasattr(batch, attr):
                    groups.extend(str(g) for g in getattr(batch, attr))
                    break
            else:
                groups.extend(str(i) for i in range(y_hat.shape[0]))
    return compute_metrics(
        torch.cat(ys), torch.cat(preds), targets, groups=groups or None
    )


def _build_trial_logger(trial, trial_config, args, study_name):
    """Per-trial logger: W&B (nice names) when requested, else CSV, else none."""
    if args.wandb_project:
        _ensure_wandb_auth()
        from lightning.pytorch.loggers import WandbLogger

        name = f"trial-{trial.number:03d}-{_make_run_name(trial_config)}"
        return WandbLogger(
            project=args.wandb_project,
            name=name,
            group=study_name,
            tags=[f"trial-{trial.number}", args.objective],
        )
    if args.log_trials_csv:
        from lightning.pytorch.loggers import CSVLogger

        return CSVLogger(
            save_dir=os.path.join(args.outdir, "hpo", "trials"),
            name=f"trial_{trial.number}",
        )
    return False


class _PruneCallback(pl.Callback):
    """Report the objective each validation epoch so Optuna can prune trials.

    Never prunes before ``min_epochs`` epochs so the noisy early-training phase
    cannot kill a promising trial (the main reason pruning felt "too strict").
    """

    def __init__(self, trial, metric: str, min_epochs: int = 0) -> None:
        self.trial = trial
        self.metric = metric
        self.min_epochs = min_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.min_epochs:
            return
        value = trainer.callback_metrics.get(self.metric)
        if value is None:
            return
        self.trial.report(float(value), step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()


# --- objective ----------------------------------------------------------------
def _make_objective(base_config, search_space, args, study_name):
    base_seed = base_config["training"]["seed"]
    direction_mode = "min" if args.direction == "minimize" else "max"

    def objective(trial: optuna.Trial) -> float:
        trial_config = copy.deepcopy(base_config)
        for path, value in _suggest(search_space, trial).items():
            set_nested(trial_config, path, value)
        # Multi-target: the model head outputs one value per target property.
        targets = normalize_targets(trial_config["target"])
        trial_config["model"]["num_targets"] = len(targets)

        # GCNConv / SAGEConv have no `edge_dim`; skip edge features for them so
        # sampled trials never crash (ScalarMoleculeModel would raise).
        conv_class = trial_config["model"].get("conv_class")
        if trial_config["model"].get("use_edge_features") and conv_class in (
            "GCNConv",
            "SAGEConv",
        ):
            trial_config["model"]["use_edge_features"] = False

        # Deterministic, per-trial seed (reproducible but distinct runs).
        if args.rotate_seeds:
            seed = (base_seed + trial.number) % (2**31)
        else:
            seed = base_seed
        set_seed(seed)
        trial_config["training"]["seed"] = seed
        # Enable Tensor Cores (TF32) + silence the "Tensor Cores" warning when
        # `cuda.tensor_cores` is set (search space may tune the flag per trial).
        configure_cuda(trial_config)

        # Cap the trial budget so HPO stays fast; early stopping ends it sooner.
        trial_config["training"]["max_epochs"] = args.max_epochs
        trial_config["training"]["patience"] = args.patience

        dataset = _get_dataset(trial_config)
        _, train_loader, val_loader, _ = build_loaders(
            dataset, trial_config["training"]
        )

        target_mean = target_std = None
        if trial_config["training"].get("normalize_targets", True):
            target_mean, target_std = fit_target_scaler(train_loader, len(targets))
        model = build_model(trial_config["model"], radius=trial_config["radius"])
        system = build_module(
            model,
            trial_config["training"],
            target_tags=[sanitize_name(t) for t in targets],
            target_mean=target_mean,
            target_std=target_std,
            config=trial_config,
        )

        callbacks: list[pl.Callback] = [
            EarlyStopping(
                monitor=args.objective,
                mode=direction_mode,
                patience=args.patience,
            )
        ]
        if not args.no_prune:
            callbacks.append(
                _PruneCallback(trial, args.objective, min_epochs=args.prune_min_epochs)
            )

        logger = _build_trial_logger(trial, trial_config, args, study_name)
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=trial_config["training"]["accelerator"],
            devices=1,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=not args.no_progress_bar,
            log_every_n_steps=10,
            **gradient_clip_kwargs(trial_config["training"]),
            callbacks=callbacks,
        )
        try:
            trainer.fit(system, train_loader, val_loader)
        except optuna.exceptions.TrialPruned:
            _finalize_wandb(logger)
            raise

        # Attach the full trial config + source YAML files to the per-trial W&B
        # run (Overview -> Config + Files tab).
        if args.wandb_project:
            _log_config_to_wandb(trial_config)
            _log_yaml_files_to_wandb(trial_config)

        metrics = _evaluate(system, val_loader, targets)
        val_loss, val_mae, val_r2 = metrics["mse"], metrics["mae"], metrics["r2"]
        trial.set_user_attr("val_loss", val_loss)
        trial.set_user_attr("val_mae", val_mae)
        trial.set_user_attr("val_r2", val_r2)
        for i, tk in enumerate(targets):
            tag = sanitize_name(tk)
            for metric in ("mse", "mae", "rmse", "r2"):
                trial.set_user_attr(
                    f"val_{tag}_{metric}", metrics[f"target_{i}_{metric}"]
                )
        trial.set_user_attr("seed", seed)
        if args.objective == "val_mae":
            value = val_mae
        elif args.objective == "val_r2":
            value = val_r2
        else:
            value = val_loss
        _finalize_wandb(logger)
        return value

    return objective


# --- CLI ----------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for the scalar GNN trainer."
    )
    parser.add_argument(
        "--config",
        default="runs/config.yaml",
        help="Base config file (.json/.yaml/.yml).",
    )
    parser.add_argument(
        "--search-space",
        default=None,
        help="Optional YAML search-space file (see runs/search_space.yaml).",
    )
    parser.add_argument(
        "--study-name", default=None, help="Optuna study name (auto if omitted)."
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--timeout", type=float, default=None, help="Stop after N seconds (optional)."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=300, help="Per-trial epoch cap."
    )
    parser.add_argument(
        "--patience", type=int, default=30, help="Early-stopping patience per trial."
    )
    parser.add_argument(
        "--objective",
        choices=OBJECTIVE_CHOICES,
        default="val_mae",
        help="val_mae / val_loss (minimize) or val_r2 (maximize; auto-forced).",
    )
    parser.add_argument(
        "--direction",
        choices=("minimize", "maximize"),
        default="minimize",
        help="Optimization direction (auto-set to maximize for val_r2).",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="sqlite:/// URL (default: <outdir>/hpo/<study>.db).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing study with the same --study-name.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Artifact root (default: base config logging.outdir).",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Log trials + study summary to this W&B project.",
    )
    parser.add_argument(
        "--log-trials-csv",
        action="store_true",
        help="Persist per-trial CSV logs (otherwise no logger for speed).",
    )
    parser.add_argument(
        "--no-prune", action="store_true", help="Disable Optuna pruning."
    )
    parser.add_argument(
        "--prune-startup-trials",
        type=int,
        default=10,
        help="Completed trials before MedianPruner starts pruning (higher = less strict).",
    )
    parser.add_argument(
        "--prune-warmup-steps",
        type=int,
        default=30,
        help="Epochs a trial trains before the pruner may consider it (higher = less strict).",
    )
    parser.add_argument(
        "--prune-min-epochs",
        type=int,
        default=10,
        help="Never prune a trial before this many epochs (keeps noisy early epochs).",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Hide Lightning progress bars.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override base training.seed."
    )
    parser.add_argument(
        "--rotate-seeds",
        action="store_true",
        help="Turn on rotating seeds, instead of using a fixed one.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Log level: DEBUG, INFO, WARNING, ERROR (default: config logging.level).",
    )
    return parser.parse_args(argv)


# --- main ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    # val_r2 is higher-better; make the study maximize it unless the user
    # explicitly chose a direction.
    if args.objective == "val_r2" and args.direction != "maximize":
        log.info("[hpo] note: val_r2 is higher-better; setting --direction maximize")
        args.direction = "maximize"

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(_ROOT, cfg_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config not found: {cfg_path}")
    base_config = deep_merge(copy.deepcopy(DEFAULT_CONFIG), load_config(cfg_path))
    if args.seed is not None:
        base_config["training"]["seed"] = args.seed
    # Multi-target: reflect the output head size in the config (and the written
    # best_config.yaml); the objective also sets it per trial.
    base_config["model"]["num_targets"] = len(normalize_targets(base_config["target"]))
    # The radius-graph cutoff must be chosen manually; cutoff_upper defaults to it.
    require_radius(base_config)
    # Apply the resolved level (--log-level wins; else logging.level; else INFO)
    # and optional file output. Handlers were added at import time.
    configure_logging(
        level=args.log_level or base_config["logging"].get("level") or "INFO",
        log_file=base_config["logging"].get("log_file"),
    )
    set_seed(base_config["training"]["seed"])
    # Enable Tensor Cores (TF32) + silence the "Tensor Cores" warning when
    # `cuda.tensor_cores` is set.
    configure_cuda(base_config)

    outdir = args.outdir or base_config["logging"]["outdir"]
    hpo_dir = os.path.join(outdir, "hpo")
    os.makedirs(hpo_dir, exist_ok=True)

    study_name = args.study_name or f"hpo-{datetime.now():%Y%m%d-%H%M%S}"
    storage = args.storage or f"sqlite:///{os.path.join(hpo_dir, study_name)}.db"

    search_space = _load_search_space(args.search_space)

    sampler = optuna.samplers.TPESampler(
        seed=base_config["training"]["seed"], n_startup_trials=5
    )
    pruner = (
        optuna.pruners.NopPruner()
        if args.no_prune
        else optuna.pruners.MedianPruner(
            n_startup_trials=args.prune_startup_trials,
            n_warmup_steps=args.prune_warmup_steps,
        )
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.resume,
    )

    log.info(
        "[hpo] study=%s storage=%s n_trials=%d objective=%s direction=%s",
        study_name,
        storage,
        args.n_trials,
        args.objective,
        args.direction,
    )
    log.info(
        "[hpo] pruning=%s startup_trials=%d warmup_steps=%d min_epochs=%d",
        "off" if args.no_prune else "median",
        args.prune_startup_trials,
        args.prune_warmup_steps,
        args.prune_min_epochs,
    )

    objective = _make_objective(base_config, search_space, args, study_name)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1,
        show_progress_bar=True,
    )

    best = study.best_trial
    log.info("[hpo] best trial=%d %s=%.6f", best.number, args.objective, best.value)
    for key, value in best.params.items():
        log.info("      %s = %s", key.replace("__", "."), value)

    # Write the best config back as YAML for `run_training.py --config`.
    best_config = copy.deepcopy(base_config)
    for path, value in best.params.items():
        set_nested(best_config, path.replace("__", "."), value)
    # The final full-length run should use the base budget and its own seed.
    best_config["training"]["seed"] = base_config["training"]["seed"]
    best_config["training"]["max_epochs"] = base_config["training"]["max_epochs"]
    best_config["training"]["patience"] = base_config["training"]["patience"]

    import yaml

    best_path = os.path.join(hpo_dir, "best_config.yaml")
    with open(best_path, "w") as f:
        yaml.safe_dump(best_config, f, sort_keys=False)
    log.info("[hpo] best config written to %s", best_path)
    log.info(
        "[hpo] launch the full run with:\n"
        "  conda run -n torch python runs/run_training.py --config %s",
        best_path,
    )

    # Parameter importances (best-effort; needs >= 2 completed trials).
    importances = {}
    try:
        importances = optuna.importance.get_param_importances(study)
        log.info("[hpo] parameter importances:")
        for param, imp in sorted(importances.items(), key=lambda kv: -kv[1]):
            log.info("      %s: %.3f", param.replace("__", "."), imp)
    except Exception as exc:
        log.info("[hpo] (could not compute importances: %s)", exc)

    # Optional W&B study summary.
    if args.wandb_project:
        import wandb

        _ensure_wandb_auth()
        wandb.init(
            project=args.wandb_project,
            name=study_name,
            job_type="hpo",
            config={
                "n_trials": len(study.trials),
                "objective": args.objective,
                "direction": args.direction,
                "best_value": best.value,
                "best_params": best.params,
            },
        )
        # Full best config + source YAML files on the study run.
        _log_config_to_wandb(best_config)
        _log_yaml_files_to_wandb(best_config)
        wandb_log = {f"best_{args.objective}": best.value}
        for key in ("val_loss", "val_mae", "val_r2"):
            if key in best.user_attrs:
                wandb_log[f"best_{key}"] = best.user_attrs[key]
        wandb.log(wandb_log)
        if importances:
            table = wandb.Table(
                columns=["param", "importance"],
                data=[
                    [param.replace("__", "."), float(imp)]
                    for param, imp in importances.items()
                ],
            )
            wandb.log(
                {
                    "param_importances": wandb.plot.bar(
                        table,
                        "param",
                        "importance",
                        title="Parameter importances",
                    )
                }
            )
        wandb.finish()


if __name__ == "__main__":
    main()
