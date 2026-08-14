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
import re
import sys

import lightning.pytorch as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

# Configure the package logger before importing the rest of morphology_gnn so
# import-time warnings (e.g. CUDA fallback) are captured. Level comes from the
# MGN_LOG_LEVEL env var, else WARNING; it is upgraded to INFO (or --log-level)
# once the config is resolved in main().
from morphology_gnn._logging import configure_logging  # noqa: E402

configure_logging(level=os.environ.get("MGN_LOG_LEVEL"))

from morphology_gnn.data import CombinedH5MolecularDataset  # noqa: E402
from morphology_gnn.model.lightning_trainer import (  # noqa: E402
    SimpleLightningMoleculeModule,
    r2_score,
)
from morphology_gnn.model.scaler_model import ScalarMoleculeModel  # noqa: E402

# Stdlib logger for the CLI. Level/handlers come from configure_logging() above
# (see --log-level and the `logging:` config section). A stable name is used
# because a script's __name__ is "__main__". Named `log` so it never collides
# with the Lightning `logger` used in main()/_finalize_wandb().
log = logging.getLogger("morphology_gnn.runs.run_training")

# --- registries: resolve config strings to classes / functions ----------------
CONV_REGISTRY = {"GATConv": GATConv, "GCNConv": GCNConv, "SAGEConv": SAGEConv}
ACT_REGISTRY = {
    "mish": F.mish,
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
    "tanh": torch.tanh,
}
OPTIMIZER_REGISTRY = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
}
SCHEDULER_REGISTRY = {
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
}
LOSS_REGISTRY = {
    "mse_loss": F.mse_loss,
    "l1_loss": F.l1_loss,
    "smooth_l1_loss": F.smooth_l1_loss,
}
# Metric registry: string names usable in `training.extra_metrics`, e.g.
# `extra_metrics: {mae: l1_loss, rmse: mse_loss, r2: r2}`.
METRIC_REGISTRY = {
    **LOSS_REGISTRY,
    "mae": F.l1_loss,
    "rmse": lambda y_hat, y: torch.sqrt(F.mse_loss(y_hat, y)),
    "r2": r2_score,
}

# --- built-in defaults (lowest precedence) ------------------------------------
DEFAULT_CONFIG = {
    "data": ["data/2-TNATA_ams.hdf5"],
    "target": "Positive VIP",
    # Radius-graph cutoff (Angstrom). REQUIRED — there is no built-in default;
    # every run must set it manually (config file `radius:` or --radius).
    "radius": None,
    "model": {
        "hidden_dim": 128,
        "num_layers": 2,
        "heads": 8,
        "num_rbf": 50,
        "use_edge_features": False,
        "conv_class": "GATConv",
        "conv_kwargs": {},
        "atom_emb_kwargs": {},
        "rbf_kwargs": {},
        # RBF distance-embedding cutoffs (used when use_edge_features is true).
        # cutoff_upper is optional: it defaults to the radius-graph cutoff
        # (`radius`) when not set. cutoff_lower defaults to 0.0. Explicit
        # cutoff_lower / cutoff_upper inside `rbf_kwargs` take precedence over
        # these first-class knobs.
        "cutoff_lower": 0.0,
        "cutoff_upper": None,
        # Residual wrapper options: whether to wrap convs, and kwargs passed to
        # the Residual wrapper ({dropout, pre_norm, post_norm}).
        "use_residual": True,
        "residual_kwargs": {},
        # First-class normalization knob (applied inside the residual wrapper):
        # names from the residual NORM_REGISTRY (Identity, LayerNorm, BatchNorm,
        # GraphNorm, InstanceNorm) + optional norm_kwargs for the chosen class.
        "norm": None,
        "norm_kwargs": {},
        # Number of output values per graph (one per target property). Set
        # automatically from the `target` list in main()/optimize().
        "num_targets": 1,
    },
    "training": {
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "max_epochs": 300,
        "patience": 30,
        "val_frac": 0.1,
        "test_frac": 0.1,
        "seed": 0,
        "num_workers": 4,
        "accelerator": "auto",
        "optimizer_class": "Adam",
        "optimizer_kwargs": {},
        "scheduler_class": None,
        "scheduler_kwargs": {},
        "scheduler_monitor": "val_loss",
        "scheduler_interval": "epoch",
        # Cross-validation: k_folds > 1 runs K-fold CV instead of the single
        # train/val/test split. Group K-fold by molecule when the dataset has
        # several distinct molecules; repeated (shuffled) K-fold (n_repeats
        # passes) when it has a single molecule. 1 / None keeps the single split.
        "k_folds": None,
        "n_repeats": 3,
        # Standardize the target(s) (per-target mean/std fit on the training
        # split) before computing the loss; predictions are un-standardized for
        # reporting. Recommended for near-constant / large-magnitude targets to
        # avoid a systematic prediction offset. Set false to use raw targets.
        "normalize_targets": True,
    },
    "logging": {
        "outdir": "runs/artifacts",
        "wandb_project": None,
        # Optional W&B niceties. `run_name` is auto-generated from the
        # hyperparameters when left None; group/tags/notes are passed straight
        # to the WandbLogger.
        "run_name": None,
        "group": None,
        "tags": None,
        "notes": None,
        # Stdlib logging for the library + CLI (morphology_gnn._logging).
        # level: WARNING (silent) / INFO (default) / DEBUG (verbose).
        "level": "INFO",
        "log_file": None,
        # Absolute path of the resolved config file, set by resolve_config().
        # Used to attach the source YAML to W&B runs.
        "config_file": None,
    },
}

# Ergonomic CLI flags mapping to dotted config paths. Each defaults to None, so a
# flag only overrides the config when it is explicitly passed.
FLAG_DEFS = [
    ("data", "data", dict(nargs="+")),
    ("target", "target", dict(nargs="+")),
    ("radius", "radius", dict(type=float)),
    ("hidden_dim", "model.hidden_dim", dict(type=int)),
    ("num_layers", "model.num_layers", dict(type=int)),
    ("heads", "model.heads", dict(type=int)),
    ("num_rbf", "model.num_rbf", dict(type=int)),
    ("cutoff_lower", "model.cutoff_lower", dict(type=float)),
    ("cutoff_upper", "model.cutoff_upper", dict(type=float)),
    (
        "use_edge_features",
        "model.use_edge_features",
        dict(action="store_true", default=None),
    ),
    ("conv_class", "model.conv_class", dict(choices=list(CONV_REGISTRY))),
    ("batch_size", "training.batch_size", dict(type=int)),
    ("k_folds", "training.k_folds", dict(type=int)),
    ("n_repeats", "training.n_repeats", dict(type=int)),
    ("lr", "training.lr", dict(type=float)),
    ("max_epochs", "training.max_epochs", dict(type=int)),
    ("patience", "training.patience", dict(type=int)),
    ("val_frac", "training.val_frac", dict(type=float)),
    ("test_frac", "training.test_frac", dict(type=float)),
    ("seed", "training.seed", dict(type=int)),
    ("num_workers", "training.num_workers", dict(type=int)),
    ("accelerator", "training.accelerator", {}),
    ("outdir", "logging.outdir", {}),
    ("wandb_project", "logging.wandb_project", {}),
    ("run_name", "logging.run_name", {}),
    ("group", "logging.group", {}),
    ("tags", "logging.tags", {}),
    ("notes", "logging.notes", {}),
    ("log_level", "logging.level", {}),
]


# --- config helpers -----------------------------------------------------------
def load_config(path: str) -> dict:
    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    elif suffix == ".json":
        with open(path) as f:
            cfg = json.load(f)
    else:
        raise ValueError(f"config file must be .json/.yaml/.yml, got: {path}")
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping, got: {type(cfg).__name__}")
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a deep copy of ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def get_nested(config: dict, dotted: str):
    node = config
    for part in dotted.split("."):
        node = node[part]
    return node


def set_nested(config: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = config
    for part in keys[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[keys[-1]] = value


def coerce(value):
    """Best-effort typed coercion for string CLI overrides."""
    if isinstance(value, str):
        low = value.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("none", "null"):
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        if value[:1] in ("[", "{"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
    return value


def normalize_targets(target) -> list[str]:
    """Accept a single target key or a list of keys; always return a list."""
    if isinstance(target, str):
        return [target]
    return list(target)


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


# --- builders ----------------------------------------------------------------
def build_model(model_cfg: dict, radius: float | None = None) -> ScalarMoleculeModel:
    """Build a ScalarMoleculeModel from a config dict.

    ``model.cutoff_upper`` is optional: when omitted (and not in ``rbf_kwargs``)
    it defaults to ``radius`` (the radius-graph cutoff), so the RBFs and the
    graph stay consistent. Explicit ``rbf_kwargs.cutoff_upper`` entries take
    precedence over the first-class ``cutoff_upper`` knob.
    """
    model_cfg = dict(model_cfg)
    conv_class = CONV_REGISTRY[model_cfg.pop("conv_class", "GATConv")]
    heads = model_cfg.pop("heads", None)
    conv_kwargs = dict(model_cfg.pop("conv_kwargs", {}) or {})
    if conv_class is GATConv:
        if heads is not None:
            conv_kwargs.setdefault("heads", heads)
        conv_kwargs.setdefault("concat", False)
    act = model_cfg.pop("act", None)
    if isinstance(act, str):
        model_cfg["act"] = ACT_REGISTRY[act]
    # `global_aggr` may be a single aggregation name, a "A+B" string, or a list
    # of names; ScalarMoleculeModel resolves/imports them (from
    # torch_geometric.nn.aggr) and builds a MultiAggregation when several are
    # given. It is passed through via **model_cfg below.
    use_residual = model_cfg.pop("use_residual", True)
    residual_kwargs = dict(model_cfg.pop("residual_kwargs", {}) or {})
    norm = model_cfg.pop("norm", None)
    norm_kwargs = dict(model_cfg.pop("norm_kwargs", {}) or {})

    # RBF distance-embedding cutoffs: `cutoff_upper` is optional and defaults to
    # the radius-graph cutoff `radius`; `cutoff_lower` defaults to 0.0. Explicit
    # `rbf_kwargs` entries take precedence over these first-class knobs.
    rbf_kwargs = dict(model_cfg.pop("rbf_kwargs", {}) or {})
    cutoff_lower = model_cfg.pop("cutoff_lower", None)
    cutoff_upper = model_cfg.pop("cutoff_upper", None)
    if cutoff_upper is None:
        cutoff_upper = rbf_kwargs.get("cutoff_upper", radius)
    if cutoff_upper is not None:
        rbf_kwargs.setdefault("cutoff_upper", cutoff_upper)
    if cutoff_lower is not None:
        rbf_kwargs.setdefault("cutoff_lower", cutoff_lower)

    return ScalarMoleculeModel(
        conv_class=conv_class,
        conv_kwargs=conv_kwargs,
        use_residual=use_residual,
        residual_kwargs=residual_kwargs,
        norm=norm,
        norm_kwargs=norm_kwargs,
        rbf_kwargs=rbf_kwargs,
        **model_cfg,
    )


def build_module(
    model,
    train_cfg: dict,
    target_tags: list[str] | None = None,
    target_mean=None,
    target_std=None,
    config: dict | None = None,
) -> SimpleLightningMoleculeModule:
    module_keys = (
        "lr",
        "weight_decay",
        "loss_func",
        "optimizer_class",
        "optimizer_kwargs",
        "scheduler_class",
        "scheduler_kwargs",
        "scheduler_monitor",
        "scheduler_interval",
        "extra_metrics",
    )
    kw = {k: train_cfg[k] for k in module_keys if k in train_cfg}
    if isinstance(kw.get("loss_func"), str):
        kw["loss_func"] = LOSS_REGISTRY[kw["loss_func"]]
    if isinstance(kw.get("optimizer_class"), str):
        kw["optimizer_class"] = OPTIMIZER_REGISTRY[kw["optimizer_class"]]
    if isinstance(kw.get("scheduler_class"), str):
        kw["scheduler_class"] = SCHEDULER_REGISTRY[kw["scheduler_class"]]
    extra = kw.get("extra_metrics")
    if isinstance(extra, dict):
        kw["extra_metrics"] = {
            name: (METRIC_REGISTRY[fn] if isinstance(fn, str) else fn)
            for name, fn in extra.items()
        }
    if target_tags is not None:
        kw["target_tags"] = target_tags
    if target_mean is not None or target_std is not None:
        kw["target_mean"] = target_mean
        kw["target_std"] = target_std
    if config is not None:
        # Full resolved config -> stored in the module's hparams, so it persists
        # in checkpoints and is auto-logged to W&B by the WandbLogger.
        kw["config"] = config
    return SimpleLightningMoleculeModule(model, **kw)


# --- secrets ------------------------------------------------------------------
def _load_dotenv(path: str | None = None) -> None:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file (if present) into ``os.environ``.

    Uses ``python-dotenv`` when available, otherwise a small dependency-free
    parser (handles comments, blank lines, ``export`` and quoted values).
    """
    try:
        import importlib

        dotenv = importlib.import_module("dotenv")
    except ImportError:
        dotenv = None
    if dotenv is not None:
        dotenv.load_dotenv(path or ".env")
        return

    env_path = path or ".env"
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)


def _ensure_wandb_auth() -> None:
    """Make sure W&B is authenticated before creating a ``WandbLogger``.

    Checks, in order: the ``WANDB_API_KEY`` environment variable, a ``.env``
    file in the project root, and the credentials stored by ``wandb login`` in
    ``~/.netrc``.
    """
    _load_dotenv()
    if os.environ.get("WANDB_API_KEY"):
        return
    try:
        import wandb

        wandb.login()
        if not wandb.api.api_key:
            raise RuntimeError("wandb.api.api_key is empty after login")
    except Exception as exc:
        raise RuntimeError(
            "wandb_project requires W&B credentials. Do one of:\n"
            "  - run `wandb login` (stores the key in ~/.netrc), or\n"
            "  - set WANDB_API_KEY in your environment, or\n"
            "  - add WANDB_API_KEY=... to a .env file in the project root."
        ) from exc


def _make_run_name(config: dict) -> str:
    """Human-friendly, unique W&B run name derived from the resolved config.

    Example: ``GATConv-h128-l2-r6-bs32-lr0.0001-heads8-rbf50-20260805-120000``.
    Set ``logging.run_name`` (or ``--run_name``) to override.
    """
    from datetime import datetime

    model = config.get("model", {})
    training = config.get("training", {})
    conv = str(model.get("conv_class", "GATConv")).rsplit(".", 1)[-1]
    parts = [
        conv,
        f"h{model.get('hidden_dim', 128)}",
        f"l{model.get('num_layers', 2)}",
        f"r{config.get('radius', 6.0):g}",
        f"bs{training.get('batch_size', 32)}",
        f"lr{training.get('lr', 1e-4):g}",
    ]
    if model.get("heads"):
        parts.append(f"heads{model['heads']}")
    if model.get("num_rbf"):
        parts.append(f"rbf{model['num_rbf']}")
    if model.get("use_edge_features"):
        parts.append("edgefeat")
    if training.get("scheduler_class"):
        parts.append(str(training["scheduler_class"]).rsplit(".", 1)[-1])
    parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    return "-".join(parts)


def _resolve_run_name(config: dict, run_name_suffix: str = "") -> str:
    """The W&B run name (explicit ``logging.run_name`` or auto-generated).

    Shared by :func:`build_logger` and the checkpoint filename prefix so the
    two always agree exactly.
    """
    name = config["logging"].get("run_name") or _make_run_name(config)
    if run_name_suffix:
        name = f"{name}-{run_name_suffix}"
    return name


def _fs_safe(name: str) -> str:
    """Filesystem-safe version of a run name (for checkpoint filenames)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")


def _finalize_wandb(logger) -> None:
    """Explicitly mark the W&B run finished so it is never flagged as crashed.

    ``WandbLogger.finalize`` only uploads checkpoints; it does **not** call
    ``wandb.finish()`` (verified in Lightning 2.6.5), so a run whose process
    simply exits can be recorded by W&B as ``crashed``. This helper calls
    ``wandb.finish()`` and is a safe no-op for CSV / None loggers.

    The helper is deliberately exception-safe: ``main()`` calls it from a
    ``finally`` block, so raising here would mask the run's real outcome (and
    could itself mark the run as crashed). Genuine failures -- e.g. ``wandb``
    unavailable, or ``wandb.finish()`` unable to flush -- are logged as
    warnings instead of being silently swallowed.
    """
    if logger is None:
        return  # Nothing was built; nothing to finish.

    try:
        from lightning.pytorch.loggers import WandbLogger
    except Exception as exc:  # pragma: no cover - lightning is imported above
        log.warning("Could not import WandbLogger while finalizing the run: %s", exc)
        return
    if not isinstance(logger, WandbLogger):
        return  # CSVLogger / other loggers have no run to finish.

    try:
        import wandb
    except Exception as exc:  # pragma: no cover - wandb is a hard dep
        log.warning("wandb is not importable while finalizing the run: %s", exc)
        return

    if wandb.run is None:
        # Expected: the run was already finished (or never initialised).
        return

    try:
        wandb.finish()
    except Exception as exc:
        # A failed flush means W&B may still record the run as crashed; surface
        # it so the failure is visible instead of being hidden.
        log.warning(
            "Failed to finish the W&B run cleanly (it may be recorded as "
            "crashed by W&B): %s",
            exc,
        )


def _log_config_to_wandb(config: dict) -> None:
    """Store the full resolved config on the active W&B run (Overview -> Config).

    Round-tripped through JSON (same as the printed config) so every value is
    serializable; a failure is logged, never raised (it must not crash the run).
    """
    try:
        import wandb

        wandb.config.update(
            json.loads(json.dumps(config, default=str)),
            allow_val_change=True,
        )
    except Exception as exc:
        log.warning("Could not log config to W&B: %s", exc)


def _log_yaml_files_to_wandb(config: dict) -> None:
    """Upload the source YAML files to the active W&B run (Files tab).

    Attaches the resolved config file (``logging.config_file``) plus the HPO
    search-space file (``runs/search_space.yaml``, when present) to the run so
    the exact configuration is preserved alongside the metrics. A failure to
    upload a file is logged, never raised.
    """
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - wandb is a hard dep
        log.warning("Could not import wandb while uploading YAML files: %s", exc)
        return
    if wandb.run is None:
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [config.get("logging", {}).get("config_file")]
    candidates += [
        os.path.join(script_dir, name)
        for name in ("config.yaml", "search_space.yaml")
    ]
    seen, uploaded = set(), 0
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        try:
            wandb.save(path)
            uploaded += 1
        except Exception as exc:
            log.warning("Could not upload %s to W&B: %s", path, exc)
    log.info("[wandb] attached %d YAML config file(s) to the run", uploaded)


def build_logger(config: dict, run_name_suffix: str = "", run_name: str | None = None):
    logging_cfg = config["logging"]
    wandb_project = logging_cfg.get("wandb_project")
    if wandb_project:
        from lightning.pytorch.loggers import WandbLogger

        _ensure_wandb_auth()
        name = (
            run_name
            if run_name is not None
            else _resolve_run_name(config, run_name_suffix)
        )
        return WandbLogger(
            project=wandb_project,
            name=name,
            group=logging_cfg.get("group"),
            tags=logging_cfg.get("tags"),
            notes=logging_cfg.get("notes"),
        )
    return CSVLogger(save_dir=logging_cfg.get("outdir", "runs/artifacts"), name="csv")


def build_callbacks(config: dict, ckpt_dir: str, name_prefix: str = "") -> list:
    """Early-stopping + best-checkpoint callbacks (single-split and CV share these).

    ``name_prefix`` (e.g. the W&B run name) is prepended to the checkpoint
    filename so checkpoints are uniquely identifiable per run.
    """
    filename = "best-{epoch}-{val_loss:.4f}"
    if name_prefix:
        filename = f"{name_prefix}-{filename}"
    return [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config["training"]["patience"],
        ),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=ckpt_dir,
            filename=filename,
        ),
    ]


# --- data ---------------------------------------------------------------------
def build_loaders(dataset, train_cfg: dict):
    n = len(dataset)
    test_size = int(train_cfg["test_frac"] * n)
    val_size = int(train_cfg["val_frac"] * n)
    train_size = n - val_size - test_size
    generator = torch.Generator().manual_seed(train_cfg["seed"])
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    log.info(
        "Samples: total=%d train=%d val=%d test=%d", n, train_size, val_size, test_size
    )

    bs, nw = train_cfg["batch_size"], train_cfg["num_workers"]
    total_loader = DataLoader(dataset, batch_size=bs, num_workers=0)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_set, batch_size=bs, num_workers=nw)
    test_loader = DataLoader(test_set, batch_size=bs, num_workers=nw)
    return total_loader, train_loader, val_loader, test_loader


def build_loaders_from_indices(dataset, train_idx, val_idx, train_cfg: dict):
    """Build train/val loaders from explicit index arrays (cross-validation)."""
    bs, nw = train_cfg["batch_size"], train_cfg["num_workers"]
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, list(train_idx)),
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, list(val_idx)),
        batch_size=bs,
        num_workers=nw,
    )
    return train_loader, val_loader


def fit_target_scaler(loader, num_targets: int):
    """Per-target mean/std for a training loader (target standardization).

    Delegates to the dataset's ``target_mean_std`` (fitted on the training
    subset indices only) when available; otherwise falls back to iterating the
    loader. Returns ``(mean, std)`` tensors of shape ``(num_targets,)``, with
    ``std = 1`` for zero-variance columns.
    """
    base, indices = loader.dataset, None
    while isinstance(base, torch.utils.data.Subset):
        indices = base.indices
        base = base.dataset
    if hasattr(base, "target_mean_std"):
        mean, std = base.target_mean_std(indices)
        return mean.view(-1), std.view(-1)
    ys = []
    for batch in loader:
        ys.append(batch.y.view(-1, num_targets))
    y = torch.cat(ys, dim=0).float()
    mean = y.mean(dim=0)
    std = y.std(dim=0)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return mean, std


# --- plotting -----------------------------------------------------------------
def sanitize_name(name: str) -> str:
    """W&B-friendly metric key: lower-case, non-alphanumeric -> underscore."""
    return re.sub(r"\W+", "_", name.strip()).strip("_").lower()


def compute_metrics(truth, pred, targets: list[str] | None = None) -> dict:
    """Aggregate + per-target regression metrics.

    Args:
        truth, pred: Tensors of shape ``(num_graphs, num_targets)``.
        targets: Optional target names used to label per-target keys.

    Returns:
        A dict with aggregate keys (``mse``, ``mae``, ``rmse``, ``r2``) and, for
        each target ``i``, ``target_{i}_{metric}`` (plus ``target_{tag}_{metric}``
        when ``targets`` is given).
    """
    t = truth.detach().cpu()
    p = pred.detach().cpu()
    ft, fp = t.view(-1), p.view(-1)
    metrics = {
        "mse": ((ft - fp) ** 2).mean().item(),
        "mae": (ft - fp).abs().mean().item(),
        "rmse": ((ft - fp) ** 2).mean().sqrt().item(),
        "r2": r2_score(fp, ft).item(),
    }
    for i in range(t.shape[1]):
        ti, pi = t[:, i], p[:, i]
        metrics[f"target_{i}_mse"] = ((ti - pi) ** 2).mean().item()
        metrics[f"target_{i}_mae"] = (ti - pi).abs().mean().item()
        metrics[f"target_{i}_rmse"] = ((ti - pi) ** 2).mean().sqrt().item()
        metrics[f"target_{i}_r2"] = r2_score(pi, ti).item()
        if targets is not None and i < len(targets):
            tag = sanitize_name(targets[i])
            for metric in ("mse", "mae", "rmse", "r2"):
                metrics[f"target_{tag}_{metric}"] = metrics[f"target_{i}_{metric}"]
    return metrics


def predict(module, loader):
    """Return ``(y, y_hat)`` over a loader, each of shape (num_graphs, num_targets)."""
    module.eval()
    device = next(module.parameters()).device
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y_hat = module(batch)  # (num_graphs, num_targets), standardized
            # PyG flattens per-graph `y` (T,) into (B*T,) when batching; reshape
            # back to (num_graphs, num_targets) so it aligns with the model head.
            ys.append(batch.y.view_as(y_hat))
            # Un-standardize predictions into the original target units.
            preds.append(module.denormalize_targets(y_hat))
    return torch.cat(ys), torch.cat(preds)


def plot_truth_vs_pred(ax, truth, pred, title) -> None:
    """Scatter + 2D histogram + KDE, exactly as in the notebook."""
    x = truth.detach().cpu().numpy()
    y = pred.detach().cpu().numpy()
    sns.scatterplot(x=x, y=y, s=5, color=".15", ax=ax)
    sns.histplot(x=x, y=y, bins=50, pthresh=0.1, cmap="mako", ax=ax)
    sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1, ax=ax)
    ax.set_xlabel("Truth")
    ax.set_ylabel("Predicted")
    ax.set_title(title)


def save_truth_vs_pred_figure(predictions, outpath: str) -> str:
    """``predictions`` is a list of ``(name, truth, pred)`` tuples."""
    n = len(predictions)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    axes = axes.ravel()
    for ax, (name, truth, pred) in zip(axes, predictions):
        plot_truth_vs_pred(ax, truth, pred, name)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Truth vs. Predicted")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- cross-validation ---------------------------------------------------------
def _cv_splits(n, mol_ids, k_folds, n_repeats, seed):
    """Yield ``(label, train_idx, val_idx)`` tuples for cross-validation.

    * Several distinct molecules -> one Group K-fold pass (whole molecules stay
      in the same fold, avoiding train/val leakage between frames).
    * A single distinct molecule -> repeated shuffled K-fold (``n_repeats``
      passes over the samples).
    """
    import random

    distinct = sorted(set(mol_ids))
    if len(distinct) > 1:
        rng = random.Random(seed)
        groups = distinct[:]
        rng.shuffle(groups)
        chunks = [[] for _ in range(k_folds)]
        for i, g in enumerate(groups):
            chunks[i % k_folds].append(g)
        sample_of = {
            m: [i for i, m2 in enumerate(mol_ids) if m2 == m] for m in distinct
        }
        for f, chunk in enumerate(chunks):
            val = [i for m in chunk for i in sample_of[m]]
            valset = set(val)
            train = [i for i in range(n) if i not in valset]
            yield f"fold_{f}", train, val
    else:
        samples = list(range(n))
        for r in range(n_repeats):
            rng = random.Random(seed + r)
            idx = samples[:]
            rng.shuffle(idx)
            folds = [idx[f::k_folds] for f in range(k_folds)]
            for f, val in enumerate(folds):
                valset = set(val)
                train = [i for i in range(n) if i not in valset]
                yield f"repeat_{r}_fold_{f}", train, val


def _aggregate_fold_metrics(fold_metrics):
    """``fold_metrics``: list of ``(label, metrics_dict)``. Return mean/std dicts."""
    import statistics

    keys = list(fold_metrics[0][1].keys())
    means, stds = {}, {}
    for k in keys:
        vals = [m[k] for _, m in fold_metrics]
        means[k] = statistics.mean(vals)
        stds[k] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return means, stds


def _write_cv_summary(cv_dir, fold_metrics, means, stds) -> str:
    import csv

    keys = list(fold_metrics[0][1].keys())
    path = os.path.join(cv_dir, "cv_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold"] + keys)
        for label, m in fold_metrics:
            w.writerow([label] + [f"{m[k]:.6f}" for k in keys])
        w.writerow(["mean"] + [f"{means[k]:.6f}" for k in keys])
        w.writerow(["std"] + [f"{stds[k]:.6f}" for k in keys])
    log.info("[cv] summary written to %s", path)
    return path


def _run_cross_validation(config, targets, dataset, outdir) -> None:
    """Run K-fold CV, replacing the single train/val/test split.

    Group K-fold by molecule when the dataset has several distinct molecules;
    repeated (shuffled) K-fold when it has a single molecule. Each fold trains a
    fresh model and is evaluated on its held-out fold; every fold's metrics,
    truth-vs-pred plot and best checkpoint are saved, and the fold metrics are
    aggregated (mean +/- std) and logged (stdout + W&B table + CSV).
    """
    training = config["training"]
    k = int(training["k_folds"])
    n_repeats = int(training.get("n_repeats", 3) or 1)
    mol_ids = dataset.mol_ids()
    distinct = sorted(set(mol_ids))
    strategy = "group" if len(distinct) > 1 else "repeated"
    if strategy == "group" and k > len(distinct):
        raise ValueError(
            f"k_folds={k} exceeds the number of distinct molecules "
            f"({len(distinct)}); reduce k_folds or provide more molecules."
        )
    suffix = f"cv-{strategy}-k{k}"
    if strategy == "repeated":
        suffix += f"x{n_repeats}"

    run_name = _resolve_run_name(config, suffix)
    ckpt_base = _fs_safe(run_name)
    logger = build_logger(config, run_name=run_name)
    # Make the W&B run active and attach the full resolved config up front.
    if config["logging"].get("wandb_project"):
        _ = logger.experiment
        _log_config_to_wandb(config)
        _log_yaml_files_to_wandb(config)

    cv_dir = os.path.join(outdir, "cv")
    os.makedirs(cv_dir, exist_ok=True)
    n_folds = k * (n_repeats if strategy == "repeated" else 1)
    log.info(
        "[cv] %s K-fold: %d fold(s), %d sample(s), %d distinct molecule(s)",
        strategy,
        n_folds,
        len(dataset),
        len(distinct),
    )

    fold_metrics = []
    try:
        splits = list(_cv_splits(len(dataset), mol_ids, k, n_repeats, training["seed"]))
        for fi, (label, train_idx, val_idx) in enumerate(splits):
            set_seed(training["seed"] + fi)  # distinct, reproducible init per fold
            train_loader, val_loader = build_loaders_from_indices(
                dataset, train_idx, val_idx, training
            )
            target_mean = target_std = None
            if training.get("normalize_targets", True):
                target_mean, target_std = fit_target_scaler(train_loader, len(targets))
            model = build_model(config["model"], radius=config["radius"])
            system = build_module(
                model,
                training,
                target_tags=[sanitize_name(t) for t in targets],
                target_mean=target_mean,
                target_std=target_std,
                config=config,
            )
            fold_dir = os.path.join(cv_dir, label)
            os.makedirs(fold_dir, exist_ok=True)
            trainer = pl.Trainer(
                max_epochs=training["max_epochs"],
                accelerator=training["accelerator"],
                devices=1,
                log_every_n_steps=10,
                callbacks=build_callbacks(
                    config,
                    os.path.join(fold_dir, "checkpoints"),
                    name_prefix=f"{ckpt_base}-{label}",
                ),
                logger=logger,
            )
            trainer.fit(system, train_loader, val_loader)

            truth, pred = predict(system, val_loader)
            metrics = compute_metrics(truth, pred, targets)
            fold_metrics.append((label, metrics))
            plot_path = os.path.join(fold_dir, "truth_vs_pred.png")
            save_truth_vs_pred_figure(
                [(f"{label} (validation)", truth.view(-1), pred.view(-1))],
                plot_path,
            )
            log.info(
                "[cv] fold %-14s MAE=%.4f RMSE=%.4f R2=%.4f n=%d",
                label,
                metrics["mae"],
                metrics["rmse"],
                metrics["r2"],
                truth.numel(),
            )
            if config["logging"].get("wandb_project"):
                import wandb

                wandb.log(
                    {
                        f"cv/{label}/val_mae": metrics["mae"],
                        f"cv/{label}/val_rmse": metrics["rmse"],
                        f"cv/{label}/val_r2": metrics["r2"],
                        f"cv/{label}/truth_vs_pred": wandb.Image(plot_path),
                    }
                )

        means, stds = _aggregate_fold_metrics(fold_metrics)
        log.info("[cv] aggregate (mean +/- std over %d folds):", len(fold_metrics))
        for mk in ("mae", "rmse", "r2"):
            log.info("      val_%-5s %.4f +/- %.4f", mk, means[mk], stds[mk])
        _write_cv_summary(cv_dir, fold_metrics, means, stds)

        if config["logging"].get("wandb_project"):
            import wandb

            for mk in ("mse", "mae", "rmse", "r2"):
                wandb.log({f"cv/{mk}_mean": means[mk], f"cv/{mk}_std": stds[mk]})
            for i, tk in enumerate(targets):
                tag = sanitize_name(tk)
                for mk in ("mae", "rmse", "r2"):
                    wandb.log(
                        {
                            f"cv/{tag}/{mk}_mean": means[f"target_{i}_{mk}"],
                            f"cv/{tag}/{mk}_std": stds[f"target_{i}_{mk}"],
                        }
                    )
            keys = list(fold_metrics[0][1].keys())
            table = wandb.Table(
                columns=["fold"] + keys,
                data=[[label] + [m[k] for k in keys] for label, m in fold_metrics],
            )
            wandb.log({"cv/fold_metrics": table})
    finally:
        _finalize_wandb(logger)


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
