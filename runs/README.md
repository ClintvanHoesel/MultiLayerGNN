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
    --model.rbf_kwargs.cutoff_upper 6.0 \
    --training.optimizer_kwargs.betas "[0.9, 0.999]" \
    --training.scheduler_class ReduceLROnPlateau \
    --logging.wandb_project InitialGNNtrial
```

The effective (merged) config is printed at the start of every run, so runs are
reproducible. Use `--config some.json` for JSON or `some.yaml` for YAML (YAML is
recommended: it supports comments and nested structures).

Common flat flags: `--data <files...> --target --radius --hidden_dim --num_layers
--heads --num_rbf --use_edge_features --conv_class --batch_size --lr --max_epochs
--patience --val_frac --test_frac --seed --num_workers --accelerator --outdir
--wandb_project --run_name --group --tags --notes`.

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
before it is closed, and the truth-vs-predicted figure + final split MAE/RMSE
are pushed to W&B before finishing.

## Hyperparameter optimization: `runs/optimize.py`

Optuna-driven HPO built on the same config + `run_training.py` blocks. Every
**trial** samples a config (based on the base config file), trains for a capped
number of epochs with early stopping + optional pruning, and returns a
validation objective to minimize (default `val_mae`). The best config is written
as YAML ready for `run_training.py --config`.

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
--patience --objective (val_mae|val_loss) --direction (minimize|maximize)
--search-space --study-name --resume --storage --wandb-project
--log-trials-csv --no-prune --no-progress-bar --seed`.

Note: GCNConv/SAGEConv trials automatically drop `use_edge_features` (those
convs have no `edge_dim` argument), so no trial crashes.


## Config file: `runs/config.yaml`

The default config, with all options and examples — including the `model.`,
`training.` and `logging.` sections and the deep `atom_emb_kwargs`,
`rbf_kwargs`, `optimizer_kwargs` and `scheduler_kwargs` maps.

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
