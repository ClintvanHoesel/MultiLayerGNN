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
--wandb_project`.

Use `--wandb-project <name>` (or `logging.wandb_project`) to log to Weights &
Biases (requires `WANDB_API_KEY` in the environment). Without it, a `CSVLogger`
is used and artifacts (checkpoints, `truth_vs_pred.png`, csv logs) land in
`logging.outdir` (`runs/artifacts/` by default).

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
