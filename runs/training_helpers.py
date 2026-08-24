"""Shared helper functions for the scalar GNN training runner.

Extracted from ``runs/run_training.py`` so the CLI entry point stays a thin
wrapper while ``runs/run_diffusion.py`` and ``runs/optimize.py`` reuse these
building blocks (they still import them through ``run_training``).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import warnings
from typing import Protocol, Sequence, cast

import lightning.pytorch as pl
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.profilers import (
    PyTorchProfiler,
    SimpleProfiler,
    AdvancedProfiler,
)
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    SAGEConv,
    CuGraphSAGEConv,
    GATv2Conv,
    CuGraphGATConv,
    RGCNConv,
    CuGraphRGCNConv,
)

from morphology_gnn.data import (
    CombinedH5MolecularDataset,
    CombinedSCMMolecularDataset,
)
from morphology_gnn.model.envelope import (
    AbstractEnvelope,
    CosineEnvelope,
    PolynomialEnvelope,
    ENVELOPE_REGISTRY,
    resolve_envelope,
)
from morphology_gnn.model.rbf import (  # noqa: F401  (re-exported for configs)
    AbstractRBF,
    BesselRBF,
    ChebychevRBF,
    ExpNormalRBF,
    GaussianRBF,
    RBF_REGISTRY,
    resolve_rbf_class,
)
from morphology_gnn.model.lightning_trainer import (
    SimpleLightningMoleculeModule,
    r2_score,
)
from morphology_gnn.model.scaler_model import ScalarMoleculeModel

log = logging.getLogger("morphology_gnn.runs.training_helpers")

# --- registries: resolve config strings to classes / functions ----------------
CONV_REGISTRY = {
    "GATConv": GATConv,
    "CuGraphGATConv": CuGraphGATConv,
    "GATv2Conv": GATv2Conv,
    "GCNConv": GCNConv,
    "SAGEConv": SAGEConv,
    "CuGraphSAGEConv": CuGraphSAGEConv,
    "RGCNConv": RGCNConv,
    "CuGraphRGCNConv": CuGraphRGCNConv,
}
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
# Envelope registry: string names usable as `model.cutoff_fn` (or
# `model.rbf_kwargs.cutoff_fn`) to select the RBF cutoff envelope.
ENVELOPE_REGISTRY = ENVELOPE_REGISTRY

# RBF registry: string names usable as `model.rbf_kwargs.rbf_class` to select
# the radial basis distance embedding (GaussianRBF, ExpNormalRBF, BesselRBF,
# ChebychevRBF; ExpNormalSmearing aliases ExpNormalRBF). Resolved by
# `resolve_rbf_class` in build_model / build_diffusion_model (and again inside
# DistanceEmbedding as a safety net for direct library use).
RBF_REGISTRY = RBF_REGISTRY  # re-exported from morphology_gnn.model.rbf

# --- built-in defaults (lowest precedence) ------------------------------------
DEFAULT_CONFIG = {
    "data": ["data/2-TNATA_ams.hdf5"],
    "target": "Positive VIP",
    # Dataset layout: "molecular" (default) or "scm".
    #   molecular — per-frame MD *_ams.hdf5 files (CombinedH5MolecularDataset).
    #   scm       — SCM-pure per-molecule files in data/data_SCM_pure/
    #               (CombinedSCMMolecularDataset); short-name targets (HOMO, S1,
    #               ...) are resolved inside the dataset class. Select via the
    #               `dataset:` config key or `--dataset scm` (see build_dataset()).
    "dataset": "scm",
    # Radius-graph cutoff (Angstrom). REQUIRED — there is no built-in default;
    # every run must set it manually (config file `radius:` or --radius).
    "radius": None,
    # Keep the SCM HDF5 data resident in memory: load it once at dataset
    # construction instead of re-reading the file on every __getitem__ /
    # accessor call. Only affects `dataset: scm` (SCMMolecularDataset /
    # CombinedSCMMolecularDataset). CLI: --keep_in_memory.
    "keep_in_memory": False,
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
        # these first-class knobs. `rbf_kwargs.rbf_class` selects the radial
        # basis: a name from RBF_REGISTRY (GaussianRBF, ExpNormalRBF,
        # BesselRBF, ChebychevRBF) or a class/instance.
        "cutoff_lower": 0.0,
        "cutoff_upper": None,
        # Optional RBF cutoff envelope: a name from ENVELOPE_REGISTRY
        # (CosineEnvelope, PolynomialEnvelope), an envelope class/instance, or
        # null to keep each RBF's own default. `model.rbf_kwargs.cutoff_fn`
        # overrides this; an explicit `rbf_kwargs.cutoff_fn: null` disables the
        # cutoff entirely (RBF features are multiplied by 1).
        "cutoff_fn": None,
        # Extra constructor kwargs for the cutoff envelope (e.g. Polynomial
        # Envelope's `exponent`). Deep `rbf_kwargs.cutoff_fn_kwargs` wins.
        "cutoff_fn_kwargs": {},
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
        # Gradient clipping (PyTorch Lightning Trainer kwargs): maximum allowed
        # gradient norm/value. 0 or None disables clipping. algorithm: "norm"
        # (default, clip by global norm) or "value" (clip each grad by value).
        "gradient_clip_val": None,
        "gradient_clip_algorithm": "norm",
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
    # CUDA / hardware knobs.
    "cuda": {
        # Use NVIDIA Tensor Cores for float32 matmuls (TF32) and silence the
        # "You are using a CUDA device ... that has Tensor Cores" warning.
        # Trades a little float32 precision for a large speedup on GPUs with
        # Tensor Cores (e.g. RTX 20xx and newer).
        "tensor_cores": False,
    },
    # Profiling (optional): `kind` selects a Lightning profiler for a training
    # run. null (default) disables profiling; "simple" records per-stage wall
    # time (data loading vs forward vs backward); "torch" wraps torch.profiler
    # with a step schedule (warmup/active) and writes chrome traces under
    # `<logging.outdir>/<run>/profile/` plus a printed top-op table. Set via
    # config `profiling.kind` or the `--profile simple|torch` CLI flag.
    "profiling": {
        "kind": None,
        # torch.profiler schedule: skip `warmup` steps, then record `active`
        # steps (see torch.profiler.schedule). Raise `warmup` to skip past
        # one-time setup overhead; lower `active` for a shorter trace.
        "warmup": 2,
        "active": 10,
        "repeat": 1,
        # Extra torch.profiler options (more detail, more overhead).
        "profile_memory": False,
        "record_shapes": False,
        "with_stack": False,
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
    ("dataset", "dataset", dict(choices=["molecular", "scm"])),
    ("radius", "radius", dict(type=float)),
    (
        "keep_in_memory",
        "keep_in_memory",
        dict(action="store_true", default=None),
    ),
    ("hidden_dim", "model.hidden_dim", dict(type=int)),
    ("num_layers", "model.num_layers", dict(type=int)),
    ("heads", "model.heads", dict(type=int)),
    ("num_rbf", "model.num_rbf", dict(type=int)),
    ("cutoff_lower", "model.cutoff_lower", dict(type=float)),
    ("cutoff_upper", "model.cutoff_upper", dict(type=float)),
    ("cutoff_fn", "model.cutoff_fn", {}),
    (
        "use_edge_features",
        "model.use_edge_features",
        dict(action="store_true", default=None),
    ),
    (
        "tensor_cores",
        "cuda.tensor_cores",
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
    ("profile", "profiling.kind", dict(choices=["simple", "torch", "advanced"])),
    ("gradient_clip_val", "training.gradient_clip_val", dict(type=float)),
    (
        "gradient_clip_algorithm",
        "training.gradient_clip_algorithm",
        dict(choices=["norm", "value"]),
    ),
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


def build_dataset(config: dict):
    """Build the scalar-regression dataset from the resolved config.

    ``dataset: molecular`` (default) -> per-frame MD files
    (:class:`CombinedH5MolecularDataset`); ``dataset: scm`` -> SCM-pure
    per-molecule files (:class:`CombinedSCMMolecularDataset`). Short-name SCM
    targets (``HOMO``, ``S1``, ...) are resolved inside the dataset class.
    """
    data_files = config["data"]
    if isinstance(data_files, str):
        data_files = [data_files]
    targets = normalize_targets(config["target"])
    if config.get("dataset") == "scm":
        return CombinedSCMMolecularDataset(
            data_files,
            targets,
            radius=config["radius"],
            keep_in_memory=config.get("keep_in_memory", False),
        )
    return CombinedH5MolecularDataset(data_files, targets, radius=config["radius"])


# --- builders ----------------------------------------------------------------
def _resolve_envelope(name) -> type[AbstractEnvelope] | AbstractEnvelope | None:
    """Resolve a config value to an envelope class/instance (``None`` stays ``None``).

    Thin wrapper over :func:`morphology_gnn.model.envelope.resolve_envelope`:
    accepts an ``AbstractEnvelope`` subclass, an instance, or a registry name
    from :data:`ENVELOPE_REGISTRY` (e.g. ``"CosineEnvelope"``). Unknown strings
    raise ``ValueError``.
    """
    return resolve_envelope(name)


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
    cutoff_fn = model_cfg.pop("cutoff_fn", None)
    cutoff_fn_kwargs = model_cfg.pop("cutoff_fn_kwargs", None)
    if cutoff_upper is None:
        cutoff_upper = rbf_kwargs.get("cutoff_upper", radius)
    if cutoff_upper is not None:
        rbf_kwargs.setdefault("cutoff_upper", cutoff_upper)
    if cutoff_lower is not None:
        rbf_kwargs.setdefault("cutoff_lower", cutoff_lower)
    # Optional RBF cutoff envelope: first-class `model.cutoff_fn`
    # (name/class/instance) or an explicit `model.rbf_kwargs.cutoff_fn` deep
    # override (which wins). Leaving it unset keeps each RBF's own default; an
    # explicit `rbf_kwargs.cutoff_fn: null` disables the cutoff (multiply by 1).
    if "cutoff_fn" not in rbf_kwargs and cutoff_fn is not None:
        rbf_kwargs["cutoff_fn"] = cutoff_fn
    if "cutoff_fn" in rbf_kwargs:
        rbf_kwargs["cutoff_fn"] = _resolve_envelope(rbf_kwargs["cutoff_fn"])
    # Envelope constructor kwargs (e.g. PolynomialEnvelope `exponent`): merge
    # first-class `model.cutoff_fn_kwargs` with deep `rbf_kwargs.cutoff_fn_kwargs`
    # (deep wins), and let the RBF build the envelope with them.
    deep_cfk = dict(rbf_kwargs.pop("cutoff_fn_kwargs", {}) or {})
    merged_cfk = dict(cutoff_fn_kwargs or {})
    merged_cfk.update(deep_cfk)
    if merged_cfk:
        rbf_kwargs["cutoff_fn_kwargs"] = merged_cfk
    # Optional RBF class: `model.rbf_kwargs.rbf_class` may be a name from
    # RBF_REGISTRY, a class, or an instance. Resolve names early so a typo
    # raises a clear config error instead of failing deep in construction.
    if "rbf_class" in rbf_kwargs:
        rbf_kwargs["rbf_class"] = resolve_rbf_class(rbf_kwargs["rbf_class"])

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


def resolve_run_outdir(config: dict, run_name: str | None = None) -> str:
    """Give every run its own subfolder under ``logging.outdir``.

    Resolves the run name (``run_name`` if given, else ``logging.run_name``,
    else the auto-generated name), fixes it in the config so it is identical
    everywhere (per-run folder, W&B run name, checkpoint filename prefix), and
    returns the per-run outdir ``<base outdir>/<run name>``. The per-run path is
    written back to ``config["logging"]["outdir"]`` so the CSV logger,
    checkpoints, figures and CV results all land inside the run's own folder.
    Call it once at the start of ``main()`` (after the config is fully resolved)
    so the folder name and the W&B/checkpoint names always agree.

    If a folder for that run name already exists (e.g. two runs auto-named in
    the same second, or re-running an explicit ``--run_name``), a numeric suffix
    (``__2``, ``__3``, ...) is appended to the *folder* so runs are never
    overwritten; the run name itself is left unchanged.
    """
    if run_name is None:
        run_name = config["logging"].get("run_name") or _resolve_run_name(config)
    config["logging"]["run_name"] = run_name
    outdir = os.path.join(config["logging"]["outdir"], _fs_safe(run_name))
    base = outdir
    n = 2
    while os.path.exists(outdir):
        outdir = f"{base}__{n}"
        n += 1
    config["logging"]["outdir"] = outdir
    os.makedirs(outdir, exist_ok=True)
    return outdir


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
        os.path.join(script_dir, name) for name in ("config.yaml", "search_space.yaml")
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


def gradient_clip_kwargs(config: dict) -> dict:
    """PyTorch Lightning Trainer kwargs for gradient clipping.

    Returns ``{}`` when clipping is disabled (``gradient_clip_val`` falsy/None),
    else ``{"gradient_clip_val": ..., "gradient_clip_algorithm": ...}`` — pass
    straight to ``pl.Trainer(**gradient_clip_kwargs(train_cfg))``. algorithm is
    ``"norm"`` (default, global-norm clip) or ``"value"`` (per-param clip).
    """
    val = config.get("gradient_clip_val")
    if not val:
        return {}
    return {
        "gradient_clip_val": float(val),
        "gradient_clip_algorithm": config.get("gradient_clip_algorithm", "norm"),
    }


def build_callbacks(
    config: dict, ckpt_dir: str, name_prefix: str = "", monitor: str = "val_loss"
) -> list:
    """Early-stopping + best-checkpoint callbacks (single-split and CV share these).

    ``name_prefix`` (e.g. the W&B run name) is prepended to the checkpoint
    filename so checkpoints are uniquely identifiable per run. ``monitor``
    defaults to ``val_loss``; pass ``train_loss`` when the run has no validation
    loader (e.g. SCM diffusion on a single box).
    """
    filename = f"best-{{epoch}}-{{{monitor}:.4f}}"
    if name_prefix:
        filename = f"{name_prefix}-{filename}"
    return [
        EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=config["training"]["patience"],
        ),
        ModelCheckpoint(
            monitor=monitor,
            mode="min",
            save_top_k=1,
            dirpath=ckpt_dir,
            filename=filename,
        ),
    ]


def build_profiler(config: dict, outdir: str):
    """Build a Lightning profiler from the ``profiling`` config section.

    Returns ``None`` when profiling is disabled (the default,
    ``profiling.kind`` is null). ``profiling.kind``:
      - ``"simple"`` -> :class:`pl.profilers.SimpleProfiler` (``extended=True``):
        per-stage wall-clock totals (data loading vs ``training_step`` vs
        ``forward`` vs ``backward``), printed to stdout at the end of
        ``trainer.fit``.
      - ``"torch"`` -> :class:`pl.profilers.PyTorchProfiler` wrapping
        ``torch.profiler`` with a step schedule (skip ``warmup`` steps, record
        ``active`` steps). On each recorded step a chrome trace is written to
        ``<outdir>/profile/`` and a top-op table (sorted by CUDA time) is
        printed — the raw material for finding GPU/CPU bottlenecks.
    """
    prof_cfg = config.get("profiling") or {}
    kind = prof_cfg.get("kind")
    if kind is None or kind is False or str(kind).lower() in ("null", "none", "false"):
        return None
    if kind == "simple":
        return SimpleProfiler(extended=True)
    if kind == "advanced":
        return AdvancedProfiler()
    if kind != "torch":
        raise ValueError(f"profiling.kind must be 'simple' or 'torch', got: {kind!r}")
    import torch.profiler as torch_profiler

    prof_dir = os.path.join(outdir, "profile")
    os.makedirs(prof_dir, exist_ok=True)
    schedule = torch_profiler.schedule(
        wait=int(prof_cfg.get("warmup", 2) or 0),
        warmup=1,
        active=int(prof_cfg.get("active", 10) or 1),
        repeat=int(prof_cfg.get("repeat", 1) or 1),
    )

    def _on_trace_ready(prof):
        path = os.path.join(prof_dir, f"trace-{prof.step_num}.json")
        try:
            prof.export_chrome_trace(path)
        except Exception as exc:  # pragma: no cover - best-effort export
            log.warning("[profiler] could not export chrome trace: %s", exc)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    return PyTorchProfiler(
        dirpath=prof_dir,
        schedule=schedule,
        on_trace_ready=_on_trace_ready,
        record_shapes=bool(prof_cfg.get("record_shapes", False)),
        profile_memory=bool(prof_cfg.get("profile_memory", False)),
        with_stack=bool(prof_cfg.get("with_stack", False)),
        with_flops=False,
    )


# --- data ---------------------------------------------------------------------
def build_loaders(dataset: Dataset, train_cfg: dict):
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
    # PyG's DataLoader type only accepts Dataset/Sequence[BaseData]/DatasetAdapter,
    # so the torch.utils.data.Subset returned by random_split is cast (runtime no-op).
    train_loader = DataLoader(
        cast(Dataset, train_set), batch_size=bs, shuffle=True, num_workers=nw
    )
    val_loader = DataLoader(cast(Dataset, val_set), batch_size=bs, num_workers=nw)
    test_loader = DataLoader(cast(Dataset, test_set), batch_size=bs, num_workers=nw)
    return total_loader, train_loader, val_loader, test_loader


def build_loaders_from_indices(dataset: Dataset, train_idx, val_idx, train_cfg: dict):
    """Build train/val loaders from explicit index arrays (cross-validation)."""
    bs, nw = train_cfg["batch_size"], train_cfg["num_workers"]
    train_loader = DataLoader(
        cast(Dataset, torch.utils.data.Subset(dataset, list(train_idx))),
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
    )
    val_loader = DataLoader(
        cast(Dataset, torch.utils.data.Subset(dataset, list(val_idx))),
        batch_size=bs,
        num_workers=nw,
    )
    return train_loader, val_loader


class _HasTargetMeanStd(Protocol):
    """Structural type for datasets exposing a fitted ``target_mean_std``."""

    def target_mean_std(
        self, indices: Sequence[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


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
        mean, std = cast(_HasTargetMeanStd, base).target_mean_std(indices)
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


def _grouped_r2(truth, pred, groups):
    """Per-group R2 and its sample-weighted mean (group-aware / within-group R2).

    ``truth``/``pred``: 1-D tensors of one target; ``groups``: one label per row.
    R2 is computed separately within each group, so a model that only separates
    groups (e.g. different materials with different mean HOMO values) scores ~0
    here instead of an inflated pooled R2. Returns ``(weighted_mean, {group: r2})``;
    degenerate groups (n <= 1 or zero-variance truth) contribute ``nan`` and are
    excluded from the weighted mean.
    """
    t = truth.detach().cpu().reshape(-1)
    p = pred.detach().cpu().reshape(-1)
    members: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(str(g), []).append(i)
    r2s: dict[str, float] = {}
    weights: list[tuple[int, float]] = []
    for lab, idx in members.items():
        tt, pp = t[idx], p[idx]
        if len(idx) > 1 and tt.std().item() > 1e-12:
            r2s[lab] = float(r2_score(pp, tt))  # torch-based r2_score
            weights.append((len(idx), r2s[lab]))
        else:
            r2s[lab] = float("nan")
    wmean = (
        sum(n * r for n, r in weights) / sum(n for n, _ in weights)
        if weights
        else float("nan")
    )
    return wmean, r2s


def compute_metrics(truth, pred, targets: list[str] | None = None, groups=None) -> dict:
    """Aggregate + per-target regression metrics.

    Args:
        truth, pred: Tensors of shape ``(num_graphs, num_targets)``.
        targets: Optional target names used to label per-target keys.
        groups: Optional per-graph group labels (e.g. material / species). When
            given (one label per graph), ``r2`` becomes the sample-weighted mean
            of the per-group R2s (``r2_within`` / ``r2_by_group``) so pooling
            datasets with different means does not inflate R2; the plain pooled
            R2 stays available as ``r2_pooled``.

    Returns:
        A dict with aggregate keys (``mse``, ``mae``, ``rmse``, ``r2``) and, for
        each target ``i``, ``target_{i}_{metric}`` (plus ``target_{tag}_{metric}``
        when ``targets`` is given).
    """
    t = truth.detach().cpu()
    p = pred.detach().cpu()
    ft, fp = t.view(-1), p.view(-1)
    pooled_r2 = r2_score(fp, ft).item()
    metrics = {
        "mse": ((ft - fp) ** 2).mean().item(),
        "mae": (ft - fp).abs().mean().item(),
        "rmse": ((ft - fp) ** 2).mean().sqrt().item(),
        "r2_pooled": pooled_r2,
    }
    has_groups = groups is not None and len(groups) == t.shape[0]
    if has_groups:
        r2_within, r2_by_group = _grouped_r2(ft, fp, groups)
    else:
        r2_within, r2_by_group = pooled_r2, {}
    if not (r2_within == r2_within):  # NaN -> fall back to pooled
        r2_within = pooled_r2
    metrics["r2"] = r2_within
    metrics["r2_within"] = r2_within
    metrics["r2_by_group"] = r2_by_group
    for i in range(t.shape[1]):
        ti, pi = t[:, i], p[:, i]
        metrics[f"target_{i}_mse"] = ((ti - pi) ** 2).mean().item()
        metrics[f"target_{i}_mae"] = (ti - pi).abs().mean().item()
        metrics[f"target_{i}_rmse"] = ((ti - pi) ** 2).mean().sqrt().item()
        metrics[f"target_{i}_r2_pooled"] = r2_score(pi, ti).item()
        if has_groups:
            g_w, g_b = _grouped_r2(ti, pi, groups)
            g_w = pooled_r2 if not (g_w == g_w) else g_w
            metrics[f"target_{i}_r2"] = g_w
            metrics[f"target_{i}_r2_within"] = g_w
            metrics[f"target_{i}_r2_by_group"] = g_b
        else:
            metrics[f"target_{i}_r2"] = metrics[f"target_{i}_r2_pooled"]
        if targets is not None and i < len(targets):
            tag = sanitize_name(targets[i])
            for metric in ("mse", "mae", "rmse", "r2"):
                metrics[f"target_{tag}_{metric}"] = metrics[f"target_{i}_{metric}"]
            metrics[f"target_{tag}_r2_pooled"] = metrics[f"target_{i}_r2_pooled"]
    return metrics


def restore_best_checkpoint(module, trainer) -> str:
    """Load the Trainer's best checkpoint weights into ``module`` in place.

    After ``trainer.fit`` the module still holds the *last-epoch* weights, so
    predictions / metrics / figures computed afterwards would not reflect the
    best model. This reloads the weights of the best checkpoint (lowest
    monitored loss, as saved by the ModelCheckpoint callback). Returns the
    checkpoint path, or ``""`` when none was saved (weights left untouched).
    """
    ckpt_cb = getattr(trainer, "checkpoint_callback", None)
    best_path = (
        getattr(ckpt_cb, "best_model_path", None) if ckpt_cb is not None else None
    )
    if not best_path:
        log.warning("no best checkpoint found; using in-memory (last-epoch) weights")
        return ""
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    module.load_state_dict(ckpt["state_dict"])
    log.info("[best-ckpt] restored weights from %s", best_path)
    return best_path


def predict(module, loader, group_attr: str = "species_name"):
    """Return ``(y, y_hat, groups)`` over a loader.

    ``y`` / ``y_hat`` are tensors of shape ``(num_graphs, num_targets)`` (raw
    target units). ``groups`` is a list of per-graph group labels used for
    group-aware R2: the material/species label (``species_name``) when present,
    else the molecule id (``mol_name``), else per-graph indices.
    """
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
            # Per-graph group labels: PyG batches string attrs into a list.
            labels = None
            for attr in (group_attr, "mol_name"):
                if hasattr(batch, attr):
                    labels = [str(g) for g in getattr(batch, attr)]
                    break
            if labels is None:
                labels = [str(i) for i in range(y_hat.shape[0])]
            groups.extend(labels)
    return torch.cat(ys), torch.cat(preds), groups


def plot_truth_vs_pred(ax, truth, pred, title) -> None:
    """Scatter + 2D histogram + KDE, exactly as in the notebook."""
    x = truth.detach().cpu().numpy()
    y = pred.detach().cpu().numpy()
    sns.scatterplot(x=x, y=y, s=5, color=".15", ax=ax)
    sns.histplot(x=x, y=y, bins=50, pthresh=0.1, cmap="mako", ax=ax)
    sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1, ax=ax)
    max = np.max([np.max(x), np.max(y)])
    min = np.min([np.min(x), np.min(y)])
    ax.plot([min, max], [min, max], "k--", lw=2.0)
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


def configure_cuda(config: dict) -> None:
    """Apply CUDA/torch runtime settings from the ``cuda`` config section.

    When ``cuda.tensor_cores`` is true this sets
    ``torch.set_float32_matmul_precision("high")`` so NVIDIA Tensor Cores are
    actually used (TF32) for float32 matmuls. Setting the precision to "high"
    also prevents PyTorch's "You are using a CUDA device (...) that has Tensor
    Cores" nag from being emitted in the first place — it only fires at the
    default "highest" precision.

    When false (the default) nothing is changed: precision stays at the torch
    default and the warning is left untouched. The precision is a global torch
    process setting, so this should be called once at the start of a run (the
    runners call it right after :func:`set_seed`).
    """
    if not config.get("cuda", {}).get("tensor_cores"):
        return
    try:
        torch.set_float32_matmul_precision("high")
    except Exception as exc:  # pragma: no cover - very old torch versions
        log.warning("[cuda] could not enable TF32 matmul precision: %s", exc)
        return
    # The "Tensor Cores" warning fires on the first float32 matmul; with the
    # precision set to "high" it normally never triggers, but filter it anyway
    # so a late warning (e.g. from a library that ran before this call) is
    # caught rather than spamming the log.
    # warnings.filterwarnings(
    #     "ignore",
    #     message=".*has Tensor Cores.*set `torch.set_float32_matmul_precision",
    # )
    log.info("[cuda] Tensor Cores enabled (float32 matmul precision = high)")


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
    """``fold_metrics``: list of ``(label, metrics_dict)``. Return mean/std dicts.

    Only scalar (int/float) metrics are aggregated; non-scalar entries (e.g. the
    per-group R2 dict ``r2_by_group``) are skipped.
    """
    import statistics

    keys = [
        k
        for k in fold_metrics[0][1].keys()
        if all(isinstance(m[k], (int, float)) for _, m in fold_metrics)
    ]
    means, stds = {}, {}
    for k in keys:
        vals = [m[k] for _, m in fold_metrics]
        means[k] = statistics.mean(vals)
        stds[k] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return means, stds


def _write_cv_summary(cv_dir, fold_metrics, means, stds) -> str:
    import csv
    import json

    def _fmt(v):
        if v is None:
            return ""
        return f"{v:.6f}" if isinstance(v, (int, float)) else json.dumps(v, default=str)

    keys = list(fold_metrics[0][1].keys())
    path = os.path.join(cv_dir, "cv_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold"] + keys)
        for label, m in fold_metrics:
            w.writerow([label] + [_fmt(m[k]) for k in keys])
        w.writerow(["mean"] + [_fmt(means.get(k)) for k in keys])
        w.writerow(["std"] + [_fmt(stds.get(k)) for k in keys])
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
                **gradient_clip_kwargs(training),
                callbacks=build_callbacks(
                    config,
                    os.path.join(fold_dir, "checkpoints"),
                    name_prefix=f"{ckpt_base}-{label}",
                ),
                logger=logger,
            )
            trainer.fit(system, train_loader, val_loader)

            # Evaluate the fold with the best checkpoint (not the last-epoch
            # weights that remain in memory after fit).
            restore_best_checkpoint(system, trainer)

            truth, pred, groups = predict(system, val_loader)
            metrics = compute_metrics(truth, pred, targets, groups=groups)
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
            import json as _json

            table = wandb.Table(
                columns=cast("list[str | int]", ["fold"] + keys),
                data=[
                    [label]
                    + [
                        (
                            m[k]
                            if isinstance(m[k], (int, float))
                            else _json.dumps(m[k], default=str)
                        )
                        for k in keys
                    ]
                    for label, m in fold_metrics
                ],
            )
            wandb.log({"cv/fold_metrics": table})
    finally:
        _finalize_wandb(logger)
