"""Train an SE(3)-equivariant diffusion model for molecular positions.

Config-driven runner for :class:`DiffusionMoleculeModel` /
:class:`DiffusionMoleculeModule`: trains an epsilon-prediction DDPM denoiser on
per-atom coordinates (a molecule conformation inside its periodic cell), then
generates new conformations conditioned on a few reference frames and reports
position metrics (coord RMSE, RDF mean-abs-diff, min-pair distances) with plots.

Configuration precedence (lowest to highest):
    built-in defaults < config file (``--config``, default ``runs/config_diffusion.yaml``)
    < CLI flags (``--lr 1e-4``) < deep overrides (``--model.hidden_dim 256``)

Example::

    python runs/run_diffusion.py --radius 4.0 --max_epochs 50 --sampling.steps 50
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
# import-time warnings (e.g. CUDA fallback) are captured.
from morphology_gnn._logging import configure_logging  # noqa: E402

configure_logging(level=os.environ.get("MGN_LOG_LEVEL"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from lightning.pytorch.loggers import CSVLogger  # noqa: E402
from torch_geometric.nn import GATConv  # noqa: E402

from morphology_gnn.data import (  # noqa: E402
    CombinedH5MolecularDataset,
    CombinedSCMDiffusionDataset,
)
from morphology_gnn.model.diffusion_model import DiffusionMoleculeModel  # noqa: E402
from morphology_gnn.model.diffusion_trainer import (  # noqa: E402
    DiffusionMoleculeModule,
    coord_rmse,
    min_pair_dist,
    rdf_hist,
    rdf_mad,
)
from morphology_gnn.radius_graph import unwrap_molecule  # noqa: E402

# Reuse the config/CLI/W&B plumbing from run_training.py (same directory, so the
# import works when running `python runs/run_diffusion.py`).
from run_training import (  # noqa: E402
    ACT_REGISTRY,
    CONV_REGISTRY,
    OPTIMIZER_REGISTRY,
    RBF_REGISTRY,
    SCHEDULER_REGISTRY,
    _ensure_wandb_auth,
    _finalize_wandb,
    _fs_safe,
    _log_config_to_wandb,
    _log_yaml_files_to_wandb,
    build_callbacks,
    build_loaders,
    coerce,
    configure_cuda,
    deep_merge,
    load_config,
    require_radius,
    resolve_envelope,
    resolve_rbf_class,
    set_nested,
    set_seed,
)

log = logging.getLogger("morphology_gnn.runs.run_diffusion")

# --- built-in defaults (lowest precedence) -----------------------------------
DEFAULT_CONFIG = {
    "data": ["data/2-TNATA_ams.hdf5"],
    # Dataset layout: "molecular" (per-frame MD *_ams.hdf5 files; the model
    # generates per-atom positions) or "scm" (SCM-pure boxes; the model then
    # generates the N molecule center-of-mass positions, ``molecules/position``).
    "dataset": "molecular",
    # Target property key(s). Required by the dataset loader, but UNUSED by the
    # diffusion model (no property conditioning).
    "target": "Positive VIP",
    # Radius-graph cutoff (Angstrom). REQUIRED — no built-in default.
    "radius": None,
    "model": {
        "hidden_dim": 128,
        "num_layers": 2,
        "heads": 4,
        "num_rbf": 50,
        "conv_class": "GATConv",
        "conv_kwargs": {},
        "rbf_kwargs": {},
        "cutoff_lower": 0.0,
        "cutoff_upper": None,  # optional -> defaults to `radius`
        "use_residual": True,
        "residual_kwargs": {},
        "norm": None,
        "norm_kwargs": {},
        "cell_embed_dim": 16,
        "self_term": True,
    },
    "diffusion": {"schedule": "cosine"},
    "training": {
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "max_epochs": 200,
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
    "sampling": {
        "steps": 100,
        "ddim": False,
        "eta": 0.0,
        "num_samples": 8,
        "num_reference": 4,
        "seed": 0,
    },
    "logging": {
        "outdir": "runs/artifacts_diffusion",
        "wandb_project": None,
        "run_name": None,
        "group": None,
        "tags": None,
        "notes": None,
        "level": "INFO",
        "log_file": None,
        "config_file": None,
    },
}

FLAG_DEFS = [
    ("data", "data", dict(nargs="+")),
    ("radius", "radius", dict(type=float)),
    ("dataset", "dataset", dict(choices=["molecular", "scm"])),
    ("hidden_dim", "model.hidden_dim", dict(type=int)),
    ("num_layers", "model.num_layers", dict(type=int)),
    ("heads", "model.heads", dict(type=int)),
    ("num_rbf", "model.num_rbf", dict(type=int)),
    ("cutoff_lower", "model.cutoff_lower", dict(type=float)),
    ("cutoff_upper", "model.cutoff_upper", dict(type=float)),
    ("conv_class", "model.conv_class", dict(choices=list(CONV_REGISTRY))),
    ("noise_schedule", "diffusion.schedule", dict(choices=["cosine", "linear"])),
    ("batch_size", "training.batch_size", dict(type=int)),
    ("lr", "training.lr", dict(type=float)),
    ("max_epochs", "training.max_epochs", dict(type=int)),
    ("patience", "training.patience", dict(type=int)),
    ("val_frac", "training.val_frac", dict(type=float)),
    ("test_frac", "training.test_frac", dict(type=float)),
    ("seed", "training.seed", dict(type=int)),
    ("num_workers", "training.num_workers", dict(type=int)),
    ("accelerator", "training.accelerator", {}),
    ("sampling_steps", "sampling.steps", dict(type=int)),
    ("sampling_ddim", "sampling.ddim", dict(action="store_true", default=None)),
    ("sampling_eta", "sampling.eta", dict(type=float)),
    ("num_samples", "sampling.num_samples", dict(type=int)),
    ("num_reference", "sampling.num_reference", dict(type=int)),
    ("outdir", "logging.outdir", {}),
    ("wandb_project", "logging.wandb_project", {}),
    ("run_name", "logging.run_name", {}),
    ("group", "logging.group", {}),
    ("tags", "logging.tags", {}),
    ("notes", "logging.notes", {}),
    ("log_level", "logging.level", {}),
]


# --- CLI + config (mirrors run_training.py) ----------------------------------
def parse_cli(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Train a diffusion position-generator (config-driven)."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file (.json/.yaml). Default: runs/config_diffusion.yaml",
    )
    for flag, _path, kwargs in FLAG_DEFS:
        add_kwargs = dict(kwargs)
        add_kwargs.setdefault("default", None)
        option_strings = [f"--{flag}", f"--{flag.replace('_', '-')}"]
        if option_strings[1] == option_strings[0]:
            option_strings = option_strings[:1]
        parser.add_argument(*option_strings, **add_kwargs)

    known = {flag for flag, _, _ in FLAG_DEFS} | {"config", "help"}
    dotted, kept = _extract_dotted_overrides(argv, known)
    args = parser.parse_args(kept)
    return args, dotted


def _extract_dotted_overrides(argv, known_flags):
    overrides, kept = {}, []
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
            os.path.dirname(os.path.abspath(__file__)), "config_diffusion.yaml"
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
    config["logging"]["config_file"] = os.path.abspath(cfg_path)
    return config


# --- builders ----------------------------------------------------------------
def build_diffusion_model(
    model_cfg: dict,
    radius: float | None = None,
    noise_schedule: str = "cosine",
) -> DiffusionMoleculeModel:
    """Build a DiffusionMoleculeModel from a config dict.

    ``model.cutoff_upper`` is optional: when omitted (and not in ``rbf_kwargs``)
    it defaults to ``radius`` so the RBFs and the graph stay consistent.
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
    use_residual = model_cfg.pop("use_residual", True)
    residual_kwargs = dict(model_cfg.pop("residual_kwargs", {}) or {})
    norm = model_cfg.pop("norm", None)
    norm_kwargs = dict(model_cfg.pop("norm_kwargs", {}) or {})
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
    # Optional RBF class: `model.rbf_kwargs.rbf_class` may be a name from
    # RBF_REGISTRY, a class, or an instance; resolve names early so a typo
    # raises a clear config error instead of failing deep in construction.
    if "rbf_class" in rbf_kwargs:
        rbf_kwargs["rbf_class"] = resolve_rbf_class(rbf_kwargs["rbf_class"])
    # Optional RBF cutoff envelope: first-class `model.cutoff_fn`
    # (name/class/instance) or an explicit `model.rbf_kwargs.cutoff_fn` deep
    # override (which wins). Leaving it unset keeps each RBF's own default; an
    # explicit `rbf_kwargs.cutoff_fn: null` disables the cutoff (multiply by 1).
    if "cutoff_fn" not in rbf_kwargs and cutoff_fn is not None:
        rbf_kwargs["cutoff_fn"] = cutoff_fn
    if "cutoff_fn" in rbf_kwargs:
        rbf_kwargs["cutoff_fn"] = resolve_envelope(rbf_kwargs["cutoff_fn"])
    # Envelope constructor kwargs (e.g. PolynomialEnvelope `exponent`): merge
    # first-class `model.cutoff_fn_kwargs` with deep `rbf_kwargs.cutoff_fn_kwargs`
    # (deep wins), and let the RBF build the envelope with them.
    deep_cfk = dict(rbf_kwargs.pop("cutoff_fn_kwargs", {}) or {})
    merged_cfk = dict(cutoff_fn_kwargs or {})
    merged_cfk.update(deep_cfk)
    if merged_cfk:
        rbf_kwargs["cutoff_fn_kwargs"] = merged_cfk

    return DiffusionMoleculeModel(
        conv_class=conv_class,
        conv_kwargs=conv_kwargs,
        use_residual=use_residual,
        residual_kwargs=residual_kwargs,
        norm=norm,
        norm_kwargs=norm_kwargs,
        rbf_kwargs=rbf_kwargs,
        noise_schedule=noise_schedule,
        **model_cfg,
    )


def build_diffusion_module(model, config: dict) -> DiffusionMoleculeModule:
    train_cfg = config["training"]
    sampling_cfg = config.get("sampling", {})
    module_keys = (
        "lr",
        "weight_decay",
        "optimizer_class",
        "optimizer_kwargs",
        "scheduler_class",
        "scheduler_kwargs",
        "scheduler_monitor",
        "scheduler_interval",
    )
    kw = {k: train_cfg[k] for k in module_keys if k in train_cfg}
    if isinstance(kw.get("optimizer_class"), str):
        kw["optimizer_class"] = OPTIMIZER_REGISTRY[kw["optimizer_class"]]
    if isinstance(kw.get("scheduler_class"), str):
        kw["scheduler_class"] = SCHEDULER_REGISTRY[kw["scheduler_class"]]
    kw["radius"] = config["radius"]
    kw["sample_steps"] = sampling_cfg.get("steps", 100)
    kw["sample_ddim"] = sampling_cfg.get("ddim", False)
    kw["sample_eta"] = sampling_cfg.get("eta", 0.0)
    kw["config"] = config
    return DiffusionMoleculeModule(model, **kw)


def build_dataset(config: dict):
    """Build the diffusion dataset from the resolved config.

    ``dataset: molecular`` -> per-frame MD files (:class:`CombinedH5MolecularDataset`,
    per-atom positions). ``dataset: scm`` -> SCM-pure boxes
    (:class:`CombinedSCMDiffusionDataset`, molecule center-of-mass positions). The
    target is unused by the diffusion model, so the SCM dataset gets ``None``.
    """
    data_files = config["data"]
    if isinstance(data_files, str):
        data_files = [data_files]
    if config.get("dataset") == "scm":
        return CombinedSCMDiffusionDataset(
            data_files, target_key=None, radius=config["radius"]
        )
    return CombinedH5MolecularDataset(
        data_files, config["target"], radius=config["radius"]
    )


# --- run name / logger -------------------------------------------------------
def _make_run_name(config: dict) -> str:
    from datetime import datetime

    model = config.get("model", {})
    training = config.get("training", {})
    sampling = config.get("sampling", {})
    conv = str(model.get("conv_class", "GATConv")).rsplit(".", 1)[-1]
    parts = [
        "diff",
        "scm" if config.get("dataset") == "scm" else "mol",
        conv,
        f"h{model.get('hidden_dim', 128)}",
        f"l{model.get('num_layers', 2)}",
        f"r{config.get('radius', 6.0):g}",
        f"bs{training.get('batch_size', 32)}",
        f"lr{training.get('lr', 1e-4):g}",
        config.get("diffusion", {}).get("schedule", "cosine"),
    ]
    parts.append(
        f"ddim{int(sampling.get('steps', 100))}"
        if sampling.get("ddim")
        else f"steps{int(sampling.get('steps', 100))}"
    )
    parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    return "-".join(parts)


def _resolve_run_name(config: dict, run_name_suffix: str = "") -> str:
    name = config["logging"].get("run_name") or _make_run_name(config)
    return f"{name}-{run_name_suffix}" if run_name_suffix else name


def build_logger(config: dict, run_name_suffix: str = "", run_name: str | None = None):
    logging_cfg = config["logging"]
    if logging_cfg.get("wandb_project"):
        from lightning.pytorch.loggers import WandbLogger

        _ensure_wandb_auth()
        name = (
            run_name
            if run_name is not None
            else _resolve_run_name(config, run_name_suffix)
        )
        return WandbLogger(
            project=logging_cfg["wandb_project"],
            name=name,
            group=logging_cfg.get("group"),
            tags=logging_cfg.get("tags"),
            notes=logging_cfg.get("notes"),
        )
    return CSVLogger(
        save_dir=logging_cfg.get("outdir", "runs/artifacts_diffusion"), name="csv"
    )


# --- generation evaluation + artifacts ---------------------------------------
def write_xyz(path: str, atom_types: torch.Tensor, pos: torch.Tensor) -> None:
    """Write a simple XYZ file for visualization."""
    types_np = atom_types.cpu().numpy()
    pos_np = pos.cpu().numpy()
    with open(path, "w") as f:
        f.write(f"{len(types_np)}\nGenerated\n")
        for z, p in zip(types_np, pos_np):
            f.write(f"{int(z):3d} {p[0]:10.4f} {p[1]:10.4f} {p[2]:10.4f}\n")


def write_xyz_symbols(path: str, symbols, pos: torch.Tensor) -> None:
    """Write an XYZ file from element symbols + positions."""
    pos_np = pos.cpu().numpy()
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\nGenerated\n")
        for s, p in zip(symbols, pos_np):
            f.write(f"{str(s):>3s} {p[0]:10.4f} {p[1]:10.4f} {p[2]:10.4f}\n")


def write_com_xyz(path: str, pos: torch.Tensor) -> None:
    """Write molecule center-of-mass points as an XYZ file (pseudo-element X)."""
    write_xyz_symbols(path, ["X"] * int(pos.shape[0]), pos)


def reconstruct_box_atoms(box_ref: dict, gen_com: torch.Tensor, box: torch.Tensor):
    """Place reference molecules at generated COM positions -> ``(symbols, pos)``.

    Each reference molecule's unwrapped conformation (atoms that straddle the
    periodic boundary pulled back together) is translated so its center of mass
    lands on the generated COM, then wrapped into the cell. Returns
    ``(element_symbols, positions)`` for the full reconstructed box, or
    ``(None, None)`` if the reference COMs are unavailable.
    """
    import numpy as np

    ref_com = box_ref.get("com")
    lattice = box_ref["lattice"]
    if ref_com is None:
        return None, None
    gen_com = gen_com.cpu()
    ref_com = ref_com.cpu()
    box = box.cpu()
    symbols: list[str] = []
    pos_list: list[torch.Tensor] = []
    for mi, atoms in enumerate(box_ref["atoms"]):
        xyz = np.stack([atoms["x"], atoms["y"], atoms["z"]], axis=-1).astype(np.float32)
        local = unwrap_molecule(torch.tensor(xyz, dtype=torch.float), lattice)
        shift = gen_com[mi] - ref_com[mi]
        wrapped = torch.remainder(local + shift, box)
        for s, p in zip(atoms["symbol"], wrapped.tolist()):
            symbols.append(s.decode() if isinstance(s, bytes) else str(s))
            pos_list.append(torch.tensor(p, dtype=torch.float))
    return symbols, torch.stack(pos_list) if pos_list else None


def evaluate_generation(module, loader, sampling_cfg: dict, device):
    """Generate conformations conditioned on a few reference frames.

    Returns ``(metrics, refs)``: aggregated position metrics and per-reference
    artifacts (atom types, cell, truth positions, generated structures,
    truth RDF) for saving/plotting.
    """
    import statistics

    ds = loader.dataset
    # Generation may run over a torch.utils.data.Subset (train/val/test split);
    # unwrap to the base dataset so per-molecule reference metadata (SCM) is
    # available for full-box reconstruction.
    base, subset_indices = ds, None
    while isinstance(base, torch.utils.data.Subset):
        subset_indices = base.indices
        base = base.dataset
    has_box_ref = hasattr(base, "box_reference")

    num_ref = min(int(sampling_cfg.get("num_reference", 4)), len(ds))
    num_samples = int(sampling_cfg.get("num_samples", 8))
    base_seed = sampling_cfg.get("seed", 0)

    all_rms, all_rdf_mad, all_min = [], [], []
    refs = []
    for i in range(num_ref):
        data = ds[i].to(device)
        atoms = data.x.squeeze(-1)
        cell = data.box.squeeze(0)  # (3,)
        truth = data.pos
        gen = module.sample_many(
            atoms, cell, n=num_samples, seed=(base_seed or 0) + i * 1000
        )  # (n, N, 3)

        rms = [coord_rmse(g, truth).item() for g in gen]
        truth_hist, edges = rdf_hist(truth, cell)
        rdf_mads = [
            rdf_mad(h, truth_hist).item() for h, _ in (rdf_hist(g, cell) for g in gen)
        ]
        min_ds = [min_pair_dist(g).item() for g in gen]

        all_rms += rms
        all_rdf_mad += rdf_mads
        all_min += min_ds
        ref = {
            "atoms": atoms.cpu(),
            "cell": cell.cpu(),
            "truth": truth.cpu(),
            "gen": gen.cpu(),
            "truth_hist": truth_hist.cpu(),
            "edges": edges.cpu(),
        }
        if has_box_ref:
            orig_idx = subset_indices[i] if subset_indices is not None else i
            # `base` is statically a PyG Dataset; box_reference only exists on the
            # SCM diffusion datasets (guarded by has_box_ref above).
            ref["box_ref"] = getattr(base, "box_reference")(orig_idx)
        refs.append(ref)

    metrics = {
        "coord_rmse_mean": statistics.mean(all_rms),
        "coord_rmse_std": statistics.pstdev(all_rms) if len(all_rms) > 1 else 0.0,
        "rdf_mad_mean": statistics.mean(all_rdf_mad),
        "rdf_mad_std": statistics.pstdev(all_rdf_mad) if len(all_rdf_mad) > 1 else 0.0,
        "min_pair_dist_mean": statistics.mean(all_min),
        "min_pair_dist_min": min(all_min),
        "n_structures": len(all_rms),
    }
    return metrics, refs


def plot_generation_figures(refs, outdir: str, split: str) -> str:
    """RDF overlay (truth vs generated) + RMSD / min-pair histograms."""
    r = refs[0]
    edges = r["edges"].numpy()
    centers = 0.5 * (edges[:-1] + edges[1:])
    gen_hists = torch.stack([rdf_hist(g, r["cell"])[0] for g in r["gen"]]).numpy()
    truth_hist = r["truth_hist"].numpy()
    mean_g = gen_hists.mean(axis=0)
    std_g = gen_hists.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(centers, truth_hist, "k-", lw=2, label="truth")
    ax.plot(centers, mean_g, "C0-", lw=1.5, label="generated (mean)")
    ax.fill_between(
        centers,
        (mean_g - std_g).clip(0),
        mean_g + std_g,
        color="C0",
        alpha=0.25,
        label=r"$\pm$ 1 std",
    )
    ax.set_xlabel("r ($\\AA$)")
    ax.set_ylabel("normalized count")
    ax.set_title(f"Radial distribution ({split})")
    ax.legend()

    ax = axes[1]
    rmsds = [coord_rmse(g, r2["truth"]).item() for r2 in refs for g in r2["gen"]]
    mins = [min_pair_dist(g).item() for r2 in refs for g in r2["gen"]]
    ax.hist(rmsds, bins=30, alpha=0.6, label="coord RMSE")
    ax.hist(mins, bins=30, alpha=0.6, label="min pair dist")
    ax.set_xlabel("$\\AA$")
    ax.set_ylabel("count")
    ax.set_title(f"Generated structure statistics ({split})")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(outdir, f"generation_{split}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_generated_structures(outdir: str, split: str, refs) -> None:
    """Save generated conformations as npz + a few XYZ files for visualization.

    For SCM box data (per-molecule reference available) the XYZ files hold the
    full reconstructed box (reference molecules placed at the generated COMs);
    otherwise they hold the generated center-of-mass / atomic positions.
    """
    import numpy as np

    for ri, r in enumerate(refs):
        np.savez_compressed(
            os.path.join(outdir, f"{split}_ref{ri}_generated.npz"),
            atom_types=r["atoms"].numpy(),
            cell=r["cell"].numpy(),
            truth=r["truth"].numpy(),
            generated=r["gen"].numpy(),  # (n, N, 3)
        )
    r = refs[0]
    box_ref = r.get("box_ref")
    for i, g in enumerate(r["gen"][:3]):
        path = os.path.join(outdir, f"{split}_sample_{i}.xyz")
        if box_ref is not None:
            symbols, pos = reconstruct_box_atoms(box_ref, g, r["cell"])
            if pos is not None:
                write_xyz_symbols(path, symbols, pos)
            else:
                write_com_xyz(path, g)
        else:
            write_xyz(path, r["atoms"], g)


# --- main --------------------------------------------------------------------
def main() -> None:
    args, dotted = parse_cli()
    config = resolve_config(args, dotted)
    require_radius(config)
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
        # 1. Dataset (positions + cell; the target is unused by the model).
        dataset = build_dataset(config)
        total_loader, train_loader, val_loader, test_loader = build_loaders(
            dataset, config["training"]
        )

        # 2. Model + Lightning wrapper.
        model = build_diffusion_model(
            config["model"],
            radius=config["radius"],
            noise_schedule=config.get("diffusion", {}).get("schedule", "cosine"),
        )
        module = build_diffusion_module(model, config)

        # 3. Logger + callbacks (early stopping + best checkpoint). With SCM data
        #    the val loader may be empty (few boxes); fall back to monitoring the
        #    training loss in that case.
        run_name = _resolve_run_name(config)
        logger = build_logger(config, run_name=run_name)
        has_val = len(val_loader) > 0
        callbacks = build_callbacks(
            config,
            os.path.join(outdir, "checkpoints"),
            name_prefix=_fs_safe(run_name),
            monitor="val_loss" if has_val else "train_loss",
        )
        trainer = pl.Trainer(
            max_epochs=config["training"]["max_epochs"],
            accelerator=config["training"]["accelerator"],
            devices=1,
            log_every_n_steps=10,
            callbacks=callbacks,
            logger=logger,
        )
        trainer.fit(module, train_loader, val_loader if has_val else None)
        if len(test_loader) > 0:
            trainer.test(module, test_loader)

        # 4. Generation evaluation. With SCM data there are only a few boxes per
        #    dataset, so evaluate generation on every box (total) rather than on
        #    possibly-empty val/test splits.
        device = next(module.parameters()).device
        gen_metrics = {}
        eval_loaders = [("val", val_loader), ("test", test_loader)]
        if config.get("dataset") == "scm" or all(len(l) == 0 for _, l in eval_loaders):
            eval_loaders = [("total", total_loader)]
        for split, loader in eval_loaders:
            metrics, refs = evaluate_generation(
                module, loader, config["sampling"], device
            )
            for k, v in metrics.items():
                gen_metrics[f"gen_{split}_{k}"] = v
            log.info("[gen/%s] %s", split, json.dumps(metrics, default=str))
            plot_path = plot_generation_figures(refs, outdir, split)
            save_generated_structures(outdir, split, refs)
            gen_metrics[f"gen_{split}_plot"] = plot_path
        log.info("Saved diffusion artifacts to %s", outdir)

        # 5. Push config + generation metrics/plots to W&B before finishing.
        if config["logging"].get("wandb_project"):
            import wandb

            _log_config_to_wandb(config)
            _log_yaml_files_to_wandb(config)
            wandb.log({k: v for k, v in gen_metrics.items() if not k.endswith("_plot")})
            for k, v in gen_metrics.items():
                if k.endswith("_plot"):
                    wandb.log({k: wandb.Image(v)})
    except BaseException:
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
        _finalize_wandb(logger)


if __name__ == "__main__":
    main()
