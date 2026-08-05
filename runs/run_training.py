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
import os
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
from torch_geometric.nn.aggr import (
    MaxAggregation,
    MeanAggregation,
    SumAggregation,
)

from morphology_gnn.data import CombinedH5MolecularDataset
from morphology_gnn.model.lightning_trainer import (
    SimpleLightningMoleculeModule,
)
from morphology_gnn.model.scaler_model import ScalarMoleculeModel

# --- registries: resolve config strings to classes / functions ----------------
CONV_REGISTRY = {"GATConv": GATConv, "GCNConv": GCNConv, "SAGEConv": SAGEConv}
ACT_REGISTRY = {
    "mish": F.mish,
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
    "tanh": torch.tanh,
}
AGGR_REGISTRY = {
    "MeanAggregation": MeanAggregation,
    "SumAggregation": SumAggregation,
    "MaxAggregation": MaxAggregation,
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

# --- built-in defaults (lowest precedence) ------------------------------------
DEFAULT_CONFIG = {
    "data": ["data/2-TNATA_ams.hdf5"],
    "target": "Positive VIP",
    "radius": 6.0,
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
    },
    "logging": {"outdir": "runs/artifacts", "wandb_project": None},
}

# Ergonomic CLI flags mapping to dotted config paths. Each defaults to None, so a
# flag only overrides the config when it is explicitly passed.
FLAG_DEFS = [
    ("data", "data", dict(nargs="+")),
    ("target", "target", {}),
    ("radius", "radius", dict(type=float)),
    ("hidden_dim", "model.hidden_dim", dict(type=int)),
    ("num_layers", "model.num_layers", dict(type=int)),
    ("heads", "model.heads", dict(type=int)),
    ("num_rbf", "model.num_rbf", dict(type=int)),
    (
        "use_edge_features",
        "model.use_edge_features",
        dict(action="store_true", default=None),
    ),
    ("conv_class", "model.conv_class", dict(choices=list(CONV_REGISTRY))),
    ("batch_size", "training.batch_size", dict(type=int)),
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
        parser.add_argument(f"--{flag}", **add_kwargs)

    known = {flag for flag, _, _ in FLAG_DEFS} | {"config", "help"}
    dotted, kept = _extract_dotted_overrides(argv, known)
    args = parser.parse_args(kept)
    return args, dotted


def _extract_dotted_overrides(argv, known_flags):
    """Pull out ``--a.b.c value``-style overrides so argparse never sees them."""
    overrides, kept = {}, []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--") and len(tok) > 2:
            name = tok[2:].split("=", 1)[0]
            if name not in known_flags:
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
        print(f"[config] {cfg_path} not found; using built-in defaults")

    for flag, path, _kwargs in FLAG_DEFS:
        value = getattr(args, flag)
        if value is not None:
            set_nested(config, path, value)
    for key, value in dotted.items():
        set_nested(config, key, value)
    return config


# --- builders ----------------------------------------------------------------
def build_model(model_cfg: dict) -> ScalarMoleculeModel:
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
    aggr = model_cfg.pop("global_aggr", None)
    if isinstance(aggr, str):
        model_cfg["global_aggr"] = AGGR_REGISTRY[aggr]
    return ScalarMoleculeModel(
        conv_class=conv_class, conv_kwargs=conv_kwargs, **model_cfg
    )


def build_module(model, train_cfg: dict) -> SimpleLightningMoleculeModule:
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
            name: (LOSS_REGISTRY[fn] if isinstance(fn, str) else fn)
            for name, fn in extra.items()
        }
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


def build_logger(logging_cfg: dict):
    wandb_project = logging_cfg.get("wandb_project")
    if wandb_project:
        from lightning.pytorch.loggers import WandbLogger

        _ensure_wandb_auth()
        return WandbLogger(project=wandb_project)
    return CSVLogger(save_dir=logging_cfg.get("outdir", "runs/artifacts"), name="csv")


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
    print(f"Samples: total={n} train={train_size} val={val_size} test={test_size}")

    bs, nw = train_cfg["batch_size"], train_cfg["num_workers"]
    total_loader = DataLoader(dataset, batch_size=bs, num_workers=0)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_set, batch_size=bs, num_workers=nw)
    test_loader = DataLoader(test_set, batch_size=bs, num_workers=nw)
    return total_loader, train_loader, val_loader, test_loader


# --- plotting -----------------------------------------------------------------
def predict(module, loader):
    """Return ``(y, y_hat)`` over a loader, in eval mode, on the module's device."""
    module.eval()
    device = next(module.parameters()).device
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(module(batch))
            ys.append(batch.y)
    return torch.cat(ys).view(-1), torch.cat(preds).view(-1)


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
    axes = axes.ravel() if n > 1 else [axes]
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


# --- main ---------------------------------------------------------------------
def main() -> None:
    _load_dotenv()  # expose keys from a git-ignored .env, if present
    args, dotted = parse_cli()
    config = resolve_config(args, dotted)
    print("[config]", json.dumps(config, indent=2, default=str))

    set_seed(config["training"]["seed"])
    outdir = config["logging"]["outdir"]
    os.makedirs(outdir, exist_ok=True)

    # 1. Dataset (one or several HDF5 files) and train/val/test split.
    data_files = config["data"]
    if isinstance(data_files, str):
        data_files = [data_files]
    dataset = CombinedH5MolecularDataset(
        data_files, config["target"], radius=config["radius"]
    )
    total_loader, train_loader, val_loader, test_loader = build_loaders(
        dataset, config["training"]
    )

    # 2. Model + Lightning wrapper.
    model = build_model(config["model"])
    system = build_module(model, config["training"])

    # 3. Logger + callbacks.
    logger = build_logger(config["logging"])
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config["training"]["patience"],
        ),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=os.path.join(outdir, "checkpoints"),
            filename="best-{epoch}-{val_loss:.4f}",
        ),
    ]
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
    predictions = []
    for name, loader in [
        ("Total", total_loader),
        ("Train", train_loader),
        ("Validation", val_loader),
        ("Test", test_loader),
    ]:
        truth, pred = predict(system, loader)
        mae = (truth - pred).abs().mean().item()
        rmse = ((truth - pred) ** 2).mean().sqrt().item()
        print(f"{name:>10}: MAE={mae:.4f}  RMSE={rmse:.4f}  n={len(truth)}")
        predictions.append((name, truth, pred))

    plot_path = os.path.join(outdir, "truth_vs_pred.png")
    save_truth_vs_pred_figure(predictions, plot_path)
    print(f"Saved truth-vs-predicted plot to {plot_path}")

    if config["logging"].get("wandb_project"):
        import wandb

        wandb.log({"truth_vs_pred": wandb.Image(plot_path)})


if __name__ == "__main__":
    main()
