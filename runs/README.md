# Runs

Python scripts for specific training runs, mirroring `notebooks/main.ipynb`.

## Runner: `run_training.py`

Config-driven. Loads one or several HDF5 files (`CombinedH5MolecularDataset`),
splits into train/val/test, trains a `ScalarMoleculeModel` under Lightning
(`SimpleLightningMoleculeModule`), and at the very end writes a
**truth-vs-predicted** figure (scatter + 2D histogram + KDE) for the total,
train, validation and test sets.

Configuration precedence (lowest to highest):

```
built-in defaults  <  config file  <  CLI flags  <  deep overrides
```

```bash
# Everything from the config file
python runs/run_training.py --config runs/config.yaml

# Override flat knobs with flags
python runs/run_training.py --config runs/config.yaml --lr 1e-3 --max_epochs 500

# Override ANY option, including deep kwargs, via dotted paths
python runs/run_training.py --config runs/config.yaml \
    --model.num_layers 4 \
    --model.atom_emb_kwargs.padding_idx 0 \
    --model.cutoff_upper 6.0 \
    --model.rbf_kwargs.rbf_class ExpNormalSmearing \
    --training.optimizer_kwargs.betas "[0.9, 0.999]" \
    --training.scheduler_class ReduceLROnPlateau \
    --logging.wandb_project InitialGNNtrial
```

The effective (merged) config is printed at the start of every run, so runs are
reproducible. Use `--config some.json` for JSON or `some.yaml` for YAML (YAML is
recommended: it supports comments and nested structures).

Common flat flags: `--data <files...> --target <key...> --radius --hidden_dim
--num_layers --heads --num_rbf --cutoff_lower --cutoff_upper
--use_edge_features --conv_class --batch_size --lr --k_folds --n_repeats
--max_epochs --patience --val_frac --test_frac --seed --num_workers --accelerator
--outdir --wandb_project --run_name --group --tags --notes`.

**Multi-target training:** pass more than one property to train on several at
once. The model outputs one value per target; the loss/metrics average over all
`(sample, target)` pairs, and **per-target** MAE/RMSE/R² are reported for every
split (and stored as per-trial attributes in HPO):

```bash
# CLI: two (or more) target keys
conda run -n torch python runs/run_training.py --config runs/config.yaml \
    --target "Positive VIP" "HOMO" --max_epochs 300

# Or in config.yaml:
# target: ["Positive VIP", "HOMO"]
```

`num_targets` (the model output size) is derived automatically from the target
list — no manual override needed.

**Cross-validation (K-fold):** set `--k_folds 5` (or `training.k_folds` in the
config) to replace the single train/val/test split with K-fold cross-validation.
The strategy is chosen automatically from the data:

- **Several distinct molecules → Group K-fold**: whole molecules stay in the
  same fold, so trajectory frames of one molecule never leak between train and
  validation.
- **A single molecule → repeated shuffled K-fold**: `--n_repeats 3` passes over
  the frames (labels `repeat_0_fold_0`, ...).

Each fold trains a fresh model, and its held-out fold is used for early stopping
and evaluation. Per fold, the run saves a best checkpoint
(`<outdir>/cv/<fold>/checkpoints`), a truth-vs-pred plot
(`<outdir>/cv/<fold>/truth_vs_pred.png`) and the per-fold metrics; the fold
metrics are then aggregated (**mean ± std**) and written to
`<outdir>/cv/cv_summary.csv`. With `--wandb-project`, one W&B run (name suffixed
`-cv-group-k5` / `-cv-repeated-k5x3`) logs every fold's metrics + plot, a table
of all folds, and the aggregate `cv/<metric>_mean` / `cv/<metric>_std` (plus
per-target values), with the full resolved config attached under Overview →
Config.

```bash
conda run -n torch python runs/run_training.py --config runs/config.yaml \
    --k_folds 5 --max_epochs 300
```

`k_folds` must not exceed the number of distinct molecules (the runner fails
fast with a clear error otherwise).

**Target standardization:** by default (`training.normalize_targets: true`) the
targets are standardized per property — mean/std are fit **on the training
split only** and the loss is computed on the standardized values — then every
reported metric/plot/prediction is un-standardized back to the original units.
This avoids a systematic prediction offset on near-constant or large-magnitude
targets (e.g. `Positive VIP` ~7.6 eV with std ~0.04 eV, where a big model
otherwise collapses to a wrong constant and R² explodes to ~−5000). Set
`training.normalize_targets: false` to train on raw targets.

Model stacking: every conv layer is wrapped in a `Residual` connection by
default. Disable with `--model.use_residual false`. Choose the normalization
type with `--model.norm LayerNorm` (or `Identity`, `BatchNorm`, `GraphNorm`,
`InstanceNorm` — from `torch_geometric.nn.norm`), with extra kwargs via
`--model.norm_kwargs "{track_running_stats: false}"`. Tune the wrapper with
`--model.residual_kwargs.pre_norm true`, `--model.residual_kwargs.post_norm true`
and `--model.residual_kwargs.dropout 0.1` (`--model.residual_kwargs.norm` is a
lower-level override of `--model.norm`).

Global (node→graph) aggregation is selected with `--model.global_aggr`. It
accepts a single PyG aggregator (`MeanAggregation`, `SumAggregation`,
`MaxAggregation`, ...), or **several combined** — either a YAML list or a
`"MeanAggregation+MaxAggregation"` string — which are joined with PyG's
`MultiAggregation` (outputs are concatenated, so the graph-feature width scales
with the number of aggregators). Any `torch_geometric.nn.aggr` name is imported
on demand (e.g. `AttentionalAggregation`, `SoftmaxAggregation`).

Use `--wandb-project <name>` (or `logging.wandb_project`) to log to Weights &
Biases (requires `WANDB_API_KEY` in the environment). Without it, a `CSVLogger`
is used and artifacts (checkpoints, `truth_vs_pred.png`, csv logs) land in
`logging.outdir` (`runs/artifacts/` by default).

### W&B run naming + clean finish

When `logging.wandb_project` is set, each run gets a human-friendly name
auto-generated from its hyperparameters, e.g.
`GATConv-h128-l2-r6-bs32-lr0.0001-heads8-rbf50-20260805-120000` (see
`_make_run_name` in `run_training.py`). Override with `--run_name` (or
`logging.run_name`); `--group`, `--tags`, `--notes` are passed through to the
`WandbLogger`.

The run is **explicitly finished** via `wandb.finish()` in a `finally` block
(`_finalize_wandb`). Lightning's `WandbLogger.finalize` does *not* call
`wandb.finish()`, so without this a run that simply exits can be flagged as
`crashed` by W&B even though training succeeded. Errors are logged onto the run
before it is closed, and the truth-vs-predicted figure + final split
MAE/RMSE/R² are pushed to W&B before finishing.

## Logging

The `morphology_gnn` library logs through the stdlib `logging` module
(`morphology_gnn.*` logger hierarchy) and is **silent by default** (WARNING
level). Both CLI scripts call `configure_logging()` (from
`morphology_gnn._logging`), so you control verbosity with:

```bash
# CLI level (wins over the config file)
conda run -n torch python runs/run_training.py --config runs/config.yaml \
    --max_epochs 1 --log-level DEBUG

# Or via the config file (default: INFO)
# logging:
#   level: DEBUG
#   log_file: runs/artifacts/train.log     # optional file output
```

Levels: `WARNING` (silent) · `INFO` (dataset sizes, radius-graph stats, model
config, per-split MAE/RMSE/R², artifacts) · `DEBUG` (per-sample loads, CUDA
path chosen, per-layer forward, Lightning steps).

In a notebook or your own scripts, enable library debug with:

```python
from morphology_gnn import configure_logging
configure_logging(level="DEBUG", log_file="/tmp/mgn.log")
```

Environment variables: `MGN_LOG_LEVEL` (fallback `LOG_LEVEL`) and `MGN_LOG_FILE`
are honored when `configure_logging` is called without explicit arguments —
e.g. `MGN_LOG_LEVEL=DEBUG conda run -n torch python runs/run_training.py ...`.
Import-time warnings (e.g. the silent CUDA fallback) are captured once logging
is configured at module import in the CLI scripts.

## Hyperparameter optimization: `runs/optimize.py`

Optuna-driven HPO built on the same config + `run_training.py` blocks. Every
**trial** samples a config (based on the base config file), trains for a capped
number of epochs with early stopping + optional pruning, and returns a
validation objective to minimize (default `val_mae`; `val_r2` maximizes). The
best config is written as YAML ready for `run_training.py --config`.

```bash
# Must use the torch conda environment
conda run -n torch python runs/optimize.py \
    --config runs/config.yaml --n-trials 20 --objective val_mae

# Custom search space (see runs/search_space.yaml for the format)
conda run -n torch python runs/optimize.py --config runs/config.yaml \
    --search-space runs/search_space.yaml --n-trials 30

# Resume a study + log trials/summary to W&B
conda run -n torch python runs/optimize.py --config runs/config.yaml \
    --study-name hpo-20260805 --resume --wandb-project InitialGNNtrial
```

Artifacts (sqlite study DB, `best_config.yaml`) land in
`<logging.outdir>/hpo/`. Key flags: `--n-trials --timeout --max-epochs
--patience --objective (val_mae|val_loss|val_r2) --direction (minimize|maximize)
--search-space --study-name --resume --storage --wandb-project
--log-trials-csv --no-prune --prune-startup-trials --prune-warmup-steps
--prune-min-epochs --no-progress-bar --seed`.

Pruning uses Optuna's `MedianPruner` and is deliberately **not too strict**:
by default a trial must run `--prune-startup-trials 10` completed trials and
`--prune-warmup-steps 10` epochs before the pruner activates, and the
`--prune-min-epochs 10` guard never prunes a trial before that epoch (so the
noisy early-training phase can't kill a promising trial). Raise any of these to
prune less, lower them to prune more aggressively, or pass `--no-prune` to
disable pruning entirely.

Note: GCNConv/SAGEConv trials automatically drop `use_edge_features` (those
convs have no `edge_dim` argument), so no trial crashes.


## Config file: `runs/config.yaml`

The default config, with all options and examples — including the `model.`,
`training.` and `logging.` sections, the RBF cutoffs (`model.cutoff_lower` /
`model.cutoff_upper`) and the deep `atom_emb_kwargs`, `rbf_kwargs`,
`optimizer_kwargs` and `scheduler_kwargs` maps.

Note: `radius` (radius-graph cutoff) is **required** — there is no built-in
default; every run must set it manually (in the config file or via `--radius`).
`model.cutoff_upper` (RBF cutoff) is optional and defaults to `radius` when
omitted; `run_training.py` and `optimize.py` fail fast with a clear error if
`radius` is missing.

## Creating a specific run

Copy the config file (or the command) and fix the settings for that experiment,
e.g. `configs/alq3_edgefeat.yaml`, `run_npb_gcn.sh`, ... Each run is
self-contained and reproducible. All CLI options are listed with
`python runs/run_training.py --help`.

## API keys & secrets

Never commit real keys. Recommended locations, in order of preference:

1. **W&B:** run `wandb login` once — the key is stored in `~/.netrc`, outside
   the repo. The runner picks it up automatically.
2. **`.env` file** in the project root (git-ignored): `WANDB_API_KEY=...`. The
   runner loads it at startup (no `python-dotenv` needed — it falls back to a
   tiny built-in parser). Copy `.env.example` to `.env` to get started.
3. **Environment variable:** `export WANDB_API_KEY=...` in your shell.

The runner checks these automatically whenever `logging.wandb_project` (or
`--wandb-project`) is set, and raises a clear error listing the options if no
credentials are found.
