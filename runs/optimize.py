"""Hyperparameter optimization for the scalar GNN trainer (``run_training.py``).

Drives :mod:`run_training`'s building blocks with an Optuna study. Every trial:

1. samples a hyperparameter config on top of a base config file,
2. builds a ``ScalarMoleculeModel`` + ``SimpleLightningMoleculeModule``,
3. trains for a capped number of epochs with early stopping (and optional
   Optuna pruning), and
4. returns a validation objective (default ``val_mae``) to minimize.

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
import os
import sys
from datetime import datetime

# Make the project root importable regardless of how this script is launched.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import lightning.pytorch as pl
import optuna
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from run_training import (  # noqa: E402  (sys.path fix above)
    DEFAULT_CONFIG,
    CombinedH5MolecularDataset,
    _ensure_wandb_auth,
    _finalize_wandb,
    _make_run_name,
    build_loaders,
    build_model,
    build_module,
    deep_merge,
    load_config,
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
    "model.hidden_dim": {"type": "categorical", "choices": [64, 128, 256]},
    "model.num_layers": {"type": "int", "low": 1, "high": 4},
    "model.heads": {"type": "categorical", "choices": [1, 2, 4, 8]},
    "model.num_rbf": {"type": "categorical", "choices": [25, 50, 75, 100]},
    "model.dropout": {"type": "float", "low": 0.0, "high": 0.5},
    "model.act": {
        "type": "categorical",
        "choices": ["gelu", "mish", "relu", "silu"],
    },
    "model.conv_class": {
        "type": "categorical",
        "choices": ["GATConv", "GCNConv", "SAGEConv"],
    },
    "model.use_edge_features": {
        "type": "categorical",
        "choices": [False, True],
    },
    "training.lr": {"type": "log_float", "low": 1e-5, "high": 1e-2},
    "training.batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    "training.weight_decay": {"type": "log_float", "low": 1e-6, "high": 1e-2},
    "training.optimizer_class": {
        "type": "categorical",
        "choices": ["Adam", "AdamW"],
    },
}

OBJECTIVE_CHOICES = ("val_mae", "val_loss")

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


def _get_dataset(config: dict) -> CombinedH5MolecularDataset:
    files = config["data"]
    if isinstance(files, str):
        files = [files]
    key = (tuple(files), config["target"], float(config["radius"]))
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = CombinedH5MolecularDataset(
            list(files), config["target"], radius=config["radius"]
        )
    return _DATASET_CACHE[key]


def _evaluate(module, loader) -> tuple[float, float]:
    """Mean MSE and MAE over a loader (deterministic; used as the objective)."""
    module.eval()
    device = next(module.parameters()).device
    n, mse, mae = 0, 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y_hat = module(batch).view(-1)
            y = batch.y.view(-1)
            mse += F.mse_loss(y_hat, y, reduction="sum").item()
            mae += F.l1_loss(y_hat, y, reduction="sum").item()
            n += y.numel()
    return mse / n, mae / n


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
    """Report the objective each validation epoch so Optuna can prune trials."""

    def __init__(self, trial, metric: str) -> None:
        self.trial = trial
        self.metric = metric

    def on_validation_epoch_end(self, trainer, pl_module):
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

        # GCNConv / SAGEConv have no `edge_dim`; skip edge features for them so
        # sampled trials never crash (ScalarMoleculeModel would raise).
        conv_class = trial_config["model"].get("conv_class")
        if trial_config["model"].get("use_edge_features") and conv_class in (
            "GCNConv",
            "SAGEConv",
        ):
            trial_config["model"]["use_edge_features"] = False

        # Deterministic, per-trial seed (reproducible but distinct runs).
        seed = (base_seed + trial.number) % (2**31)
        set_seed(seed)
        trial_config["training"]["seed"] = seed

        # Cap the trial budget so HPO stays fast; early stopping ends it sooner.
        trial_config["training"]["max_epochs"] = args.max_epochs
        trial_config["training"]["patience"] = args.patience

        dataset = _get_dataset(trial_config)
        _, train_loader, val_loader, _ = build_loaders(
            dataset, trial_config["training"]
        )

        model = build_model(trial_config["model"])
        system = build_module(model, trial_config["training"])

        callbacks = [
            EarlyStopping(
                monitor=args.objective,
                mode=direction_mode,
                patience=args.patience,
            )
        ]
        if not args.no_prune:
            callbacks.append(_PruneCallback(trial, args.objective))

        logger = _build_trial_logger(trial, trial_config, args, study_name)
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=trial_config["training"]["accelerator"],
            devices=1,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=not args.no_progress_bar,
            log_every_n_steps=10,
            callbacks=callbacks,
        )
        try:
            trainer.fit(system, train_loader, val_loader)
        except optuna.exceptions.TrialPruned:
            _finalize_wandb(logger)
            raise

        val_loss, val_mae = _evaluate(system, val_loader)
        trial.set_user_attr("val_loss", val_loss)
        trial.set_user_attr("val_mae", val_mae)
        trial.set_user_attr("seed", seed)
        value = val_mae if args.objective == "val_mae" else val_loss
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
        "--max-epochs", type=int, default=60, help="Per-trial epoch cap."
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Early-stopping patience per trial."
    )
    parser.add_argument("--objective", choices=OBJECTIVE_CHOICES, default="val_mae")
    parser.add_argument(
        "--direction", choices=("minimize", "maximize"), default="minimize"
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
        "--no-progress-bar",
        action="store_true",
        help="Hide Lightning progress bars.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override base training.seed."
    )
    return parser.parse_args(argv)


# --- main ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(_ROOT, cfg_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config not found: {cfg_path}")
    base_config = deep_merge(copy.deepcopy(DEFAULT_CONFIG), load_config(cfg_path))
    if args.seed is not None:
        base_config["training"]["seed"] = args.seed
    set_seed(base_config["training"]["seed"])

    outdir = args.outdir or base_config["logging"]["outdir"]
    hpo_dir = os.path.join(outdir, "hpo")
    os.makedirs(hpo_dir, exist_ok=True)

    study_name = args.study_name or f"hpo-{datetime.now():%Y%m%d-%H%M%S}"
    storage = args.storage or f"sqlite:///{os.path.join(hpo_dir, study_name)}.db"

    search_space = _load_search_space(args.search_space)

    sampler = optuna.samplers.TPESampler(
        seed=base_config["training"]["seed"], n_startup_trials=5
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.resume,
    )

    print(
        f"[hpo] study={study_name} storage={storage} "
        f"n_trials={args.n_trials} objective={args.objective} "
        f"direction={args.direction}"
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
    print(f"[hpo] best trial={best.number} {args.objective}={best.value:.6f}")
    for key, value in best.params.items():
        print(f"      {key.replace('__', '.')} = {value}")

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
    print(f"[hpo] best config written to {best_path}")
    print(
        "[hpo] launch the full run with:\n"
        f"  conda run -n torch python runs/run_training.py --config {best_path}"
    )

    # Parameter importances (best-effort; needs >= 2 completed trials).
    importances = {}
    try:
        importances = optuna.importance.get_param_importances(study)
        print("[hpo] parameter importances:")
        for param, imp in sorted(importances.items(), key=lambda kv: -kv[1]):
            print(f"      {param.replace('__', '.')}: {imp:.3f}")
    except Exception as exc:
        print(f"[hpo] (could not compute importances: {exc})")

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
        log = {f"best_{args.objective}": best.value}
        for key in ("val_loss", "val_mae"):
            if key in best.user_attrs:
                log[f"best_{key}"] = best.user_attrs[key]
        wandb.log(log)
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
