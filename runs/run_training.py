"""Train a scalar-regression GNN on molecular HDF5 data, driven by a config file.

Mirrors the flow of ``notebooks/main.ipynb`` but uses the modular
``morphology_gnn`` model stack, adds a train/val/test split, and finishes with a
truth-vs-predicted figure (scatter + density histogram + KDE) for the total,
train, validation and test sets.

Configuration precedence (lowest to highest):
    built-in defaults < config file (``--config``, default ``runs/config.yaml``)
    < CLI flags (``--lr 1e-4``) < deep overrides (``--model.rbf_kwargs.cutoff_upper 6.0``)

Any option can be overridden from the command line, including nested ones, by
passing the dotted config path::

    python runs/run_training.py \\
        --model.num_layers 4 \\
        --model.atom_emb_kwargs.padding_idx 0 \\
        --model.rbf_kwargs.cutoff_upper 6.0 \\
        --training.optimizer_kwargs.betas "[0.9, 0.999]" \\
        --training.scheduler_class ReduceLROnPlateau \\
        --logging.wandb_project InitialGNNtrial
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys

import lightning.pytorch as pl

# Configure the package logger before importing the rest of morphology_gnn so
# import-time warnings (e.g. CUDA fallback) are captured. Level comes from the
# MGN_LOG_LEVEL env var, else WARNING; it is upgraded to INFO (or --log-level)
# once the config is resolved in main().
from morphology_gnn._logging import configure_logging  # noqa: E402

configure_logging(level=os.environ.get("MGN_LOG_LEVEL"))

from morphology_gnn.data import CombinedH5MolecularDataset  # noqa: E402

# Shared building blocks (registries, config/default plumbing, model builders,
# data + plotting helpers, cross-validation and W&B utilities) now live in
# `runs/training_helpers.py`. They are imported here for ``main()`` and are
# re-exported so ``run_diffusion.py`` / ``optimize.py`` keep working unchanged.
from training_helpers import (  # noqa: E402
    ACT_REGISTRY,
    CONV_REGISTRY,
    DEFAULT_CONFIG,
    ENVELOPE_REGISTRY,
    FLAG_DEFS,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
    OPTIMIZER_REGISTRY,
    RBF_REGISTRY,
    SCHEDULER_REGISTRY,
    _ensure_wandb_auth,
    _finalize_wandb,
    _fs_safe,
    _load_dotenv,
    _log_config_to_wandb,
    _log_yaml_files_to_wandb,
    _make_run_name,
    _resolve_envelope,
    _resolve_run_name,
    resolve_envelope,
    _run_cross_validation,
    build_callbacks,
    build_loaders,
    build_loaders_from_indices,
    build_logger,
    build_model,
    build_module,
    coerce,
    compute_metrics,
    configure_cuda,
    deep_merge,
    fit_target_scaler,
    get_nested,
    load_config,
    normalize_targets,
    predict,
    resolve_rbf_class,
    sanitize_name,
    save_truth_vs_pred_figure,
    set_nested,
    set_seed,
)

# Stdlib logger for the CLI. Level/handlers come from configure_logging() above
# (see --log-level and the `logging:` config section). A stable name is used
# because a script's __name__ is "__main__". Named `log` so it never collides
# with the Lightning `logger` used in main()/_finalize_wandb().
log = logging.getLogger("morphology_gnn.runs.run_training")


# --- CLI ----------------------------------------------------------------------
def parse_cli(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Train a scalar molecular GNN (config-driven)."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file (.json/.yaml). Default: runs/config.yaml",
    )
    for flag, _path, kwargs in FLAG_DEFS:
        add_kwargs = dict(kwargs)
        add_kwargs.setdefault("default", None)
        # Register both spellings so `--max_epochs` and `--max-epochs` work.
        option_strings = [f"--{flag}", f"--{flag.replace('_', '-')}"]
        if option_strings[1] == option_strings[0]:
            option_strings = option_strings[:1]
        parser.add_argument(*option_strings, **add_kwargs)

    known = {flag for flag, _, _ in FLAG_DEFS} | {"config", "help"}
    dotted, kept = _extract_dotted_overrides(argv, known)
    args = parser.parse_args(kept)
    return args, dotted


def _extract_dotted_overrides(argv, known_flags):
    """Pull out ``--a.b.c value``-style overrides so argparse never sees them."""
    overrides, kept = {}, []
    # Normalize so hyphen/underscore flag spellings are both recognized.
    known = {name.replace("-", "_") for name in known_flags}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--") and len(tok) > 2:
            name = tok[2:].split("=", 1)[0]
            if name.replace("-", "_") not in known:
                if "=" in tok:
                    key, raw = tok[2:].split("=", 1)
                    overrides[key] = coerce(raw)
                else:
                    nxt = argv[i + 1] if i + 1 < len(argv) else None
                    if nxt is not None and not nxt.startswith("--"):
                        overrides[name] = coerce(nxt)
                        i += 1
                    else:
                        overrides[name] = True
                i += 1
                continue
        kept.append(tok)
        i += 1
    return overrides, kept


def resolve_config(args, dotted) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    cfg_path = args.config
    if cfg_path is None:
        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.yaml"
        )
    if os.path.exists(cfg_path):
        config = deep_merge(config, load_config(cfg_path))
    else:
        log.warning("[config] %s not found; using built-in defaults", cfg_path)

    for flag, path, _kwargs in FLAG_DEFS:
        value = getattr(args, flag)
        if value is not None:
            set_nested(config, path, value)
    for key, value in dotted.items():
        set_nested(config, key, value)
    # Remember the config file path so the source YAML can be attached to W&B
    # runs and included in the logged/stored config.
    config["logging"]["config_file"] = os.path.abspath(cfg_path)
    return config


def require_radius(config: dict) -> None:
    """Enforce that the radius-graph cutoff is chosen manually (no default).

    ``radius`` must always be entered explicitly (config file ``radius:`` or
    ``--radius``). ``model.cutoff_upper`` is optional and defaults to ``radius``
    when omitted (see :func:`build_model`).
    """
    if get_nested(config, "radius") is None:
        raise ValueError(
            "radius is required and has no default. Set `radius` in the config "
            "file (e.g. runs/config.yaml) or pass `--radius`."
        )


# --- main ---------------------------------------------------------------------
def main() -> None:
    _load_dotenv()  # expose keys from a git-ignored .env, if present
    args, dotted = parse_cli()
    config = resolve_config(args, dotted)
    # The radius-graph cutoff must be chosen manually; cutoff_upper defaults to it.
    require_radius(config)
    # Multi-target: derive the model output size from the target list up front so
    # the printed config reflects the effective model head.
    targets = normalize_targets(config["target"])
    config["model"]["num_targets"] = len(targets)
    # Apply the resolved level (--log-level wins; else logging.level; else INFO)
    # and optional file output. Handlers were added at import time, so this only
    # updates the level / adds a file handler.
    configure_logging(
        level=args.log_level or config["logging"].get("level") or "INFO",
        log_file=config["logging"].get("log_file"),
    )
    log.info("[config] %s", json.dumps(config, indent=2, default=str))

    set_seed(config["training"]["seed"])
    # Enable Tensor Cores (TF32) + silence the "Tensor Cores" warning when
    # `cuda.tensor_cores` is set.
    configure_cuda(config)
    outdir = config["logging"]["outdir"]
    os.makedirs(outdir, exist_ok=True)

    logger = None
    try:
        # 1. Dataset (one or several HDF5 files).
        data_files = config["data"]
        if isinstance(data_files, str):
            data_files = [data_files]
        dataset = CombinedH5MolecularDataset(
            data_files, targets, radius=config["radius"]
        )

        # Cross-validation mode (k_folds > 1) replaces the single split below.
        if config["training"].get("k_folds") and config["training"]["k_folds"] > 1:
            _run_cross_validation(config, targets, dataset, outdir)
            return

        total_loader, train_loader, val_loader, test_loader = build_loaders(
            dataset, config["training"]
        )

        # 2. Model + Lightning wrapper. Per-target labels let Lightning log one
        #    loss/metric per property ({split}_target_{tag}_{metric}). The RBF
        #    cutoff_upper defaults to the radius-graph cutoff when not set.
        #    Targets are standardized by default (fit on the train split) so the
        #    loss is well-conditioned; predictions are reported in raw units.
        target_mean = target_std = None
        if config["training"].get("normalize_targets", True):
            target_mean, target_std = fit_target_scaler(train_loader, len(targets))
            log.info(
                "[scaler] target mean=%s std=%s",
                target_mean.tolist(),
                target_std.tolist(),
            )
        model = build_model(config["model"], radius=config["radius"])
        system = build_module(
            model,
            config["training"],
            target_tags=[sanitize_name(t) for t in targets],
            target_mean=target_mean,
            target_std=target_std,
            config=config,
        )

        # 3. Logger + callbacks. Checkpoint filenames carry the (W&B) run name.
        run_name = _resolve_run_name(config)
        logger = build_logger(config, run_name=run_name)
        callbacks = build_callbacks(
            config,
            os.path.join(outdir, "checkpoints"),
            name_prefix=_fs_safe(run_name),
        )
        trainer = pl.Trainer(
            max_epochs=config["training"]["max_epochs"],
            accelerator=config["training"]["accelerator"],
            devices=1,
            log_every_n_steps=10,
            callbacks=callbacks,
            logger=logger,
        )
        trainer.fit(system, train_loader, val_loader)

        # 4. Final truth-vs-predicted figure for total / train / validation / test.
        predictions, final_metrics = [], {}
        for name, loader in [
            ("Total", total_loader),
            ("Train", train_loader),
            ("Validation", val_loader),
            ("Test", test_loader),
        ]:
            truth, pred = predict(system, loader)
            metrics = compute_metrics(truth, pred, targets)
            log.info(
                "%10s: MAE=%.4f  RMSE=%.4f  R2=%.4f  n=%d",
                name,
                metrics["mae"],
                metrics["rmse"],
                metrics["r2"],
                truth.numel(),
            )
            for i, tk in enumerate(targets):
                log.info(
                    "    [%s] MAE=%.4f  RMSE=%.4f  R2=%.4f",
                    tk,
                    metrics[f"target_{i}_mae"],
                    metrics[f"target_{i}_rmse"],
                    metrics[f"target_{i}_r2"],
                )
            predictions.append((name, truth.view(-1), pred.view(-1)))
            prefix = name.lower()
            for metric in ("mae", "rmse", "r2"):
                final_metrics[f"{prefix}_{metric}"] = metrics[metric]
            for i, tk in enumerate(targets):
                tag = sanitize_name(tk)
                for metric in ("mae", "rmse", "r2"):
                    final_metrics[f"{prefix}_{tag}_{metric}"] = metrics[
                        f"target_{i}_{metric}"
                    ]

        plot_path = os.path.join(outdir, "truth_vs_pred.png")
        save_truth_vs_pred_figure(predictions, plot_path)
        log.info("Saved truth-vs-predicted plot to %s", plot_path)

        # 5. Push the figure + final split metrics to W&B *before* finishing,
        #    so the run is recorded as "finished" (not "crashed") with all
        #    artifacts attached.
        if config["logging"].get("wandb_project"):
            import wandb

            # Full resolved config on the run (Overview -> Config) + the source
            # YAML files (Files tab), then the figure + final split metrics.
            _log_config_to_wandb(config)
            _log_yaml_files_to_wandb(config)
            wandb.log({"truth_vs_pred": wandb.Image(plot_path)})
            wandb.log(final_metrics)
    except BaseException:
        # Record the failure on the run, then re-raise (the finally block still
        # marks the run finished rather than crashed).
        if config["logging"].get("wandb_project"):
            try:
                import traceback
                import wandb

                if wandb.run is not None:
                    wandb.log({"error": traceback.format_exc()})
            except Exception:
                pass
        raise
    finally:
        # Explicitly close the W&B run (Lightning's WandbLogger does not).
        _finalize_wandb(logger)


if __name__ == "__main__":
    main()
