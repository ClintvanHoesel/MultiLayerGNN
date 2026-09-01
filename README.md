# MultiLayerGNN

THIS REPOSITORY IS A WORK IN PROGRESS.
POSTPONED (hopefully) TEMPORARILY AWAITING MORE TRAINING DATA.

This repository contains my work on a graph neural network that bridges
molecular and morphology length scales. I am interested in the range from a
few nanometres (roughly 10³ molecules) to hundreds of nanometres (up to around
10⁶ molecules).

A molecule does not have the same properties in a vacuum as it does in a real
morphology. Its neighbours, local packing, and the periodic environment can all
affect properties such as the HOMO, LUMO, ionisation potential, and excitation
energies. The aim here is to include that environment without having to model a
whole large morphology at atomistic resolution.

## Idea

I use three connected levels:

1. **Atoms.** Atoms are connected with a radius graph. Their types and
   distance-based edge features describe the molecular geometry.
2. **The local environment.** In the box data, I can add the nearby molecules
   around one query molecule. They are selected using periodic COM distances,
   either by radius, nearest neighbours, or by taking the full box. Only the
   query molecule is used for the loss; the other molecules give it context.
3. **Molecules.** The hierarchical model pools each molecule into a
   mass-weighted centre-of-mass (COM) representation, lets molecules exchange
   messages, and passes the information back to the atoms.

The idea is not to replace full simulations, but to learn a useful surrogate
for morphology-dependent molecular properties at a scale where a fully
atomistic treatment of every molecule is no longer practical.

At the moment the repository includes property prediction from HDF5 MD and box
data, multi-target training, PBC-aware graphs, the optional atomistic + COM
hierarchy, cross-validation, W&B logging, and a diffusion model for periodic
molecular geometries.

## Where things are

| Path | Purpose |
| --- | --- |
| `morphology_gnn/data.py` | HDF5 datasets, periodic graph construction, box context assembly |
| `morphology_gnn/model/` | Scalar, hierarchical, and diffusion GNN models |
| `morphology_gnn/radius_graph.py` | Periodic-boundary and COM geometry utilities |
| `runs/config.yaml` | Main property-regression configuration and documented options |
| `runs/run_training.py` | Config-driven property-training entry point |
| `runs/run_diffusion.py` | Config-driven diffusion-training entry point |
| `runs/README.md` | Detailed runner, cross-validation, and logging documentation |
| `tests/` | Unit tests for geometry, context, hierarchy, diffusion, and model pieces |

## Getting started

I use Python 3.13, PyTorch, and PyTorch Geometric. The core dependencies are
listed in `pyproject.toml`; the training scripts need the extra packages below.

```bash
git clone <repository-url>
cd MultiLayerGNN

# Install the PyTorch build that matches your CUDA/CPU setup first, then:
pip install -e .
pip install lightning h5py pyyaml matplotlib pytest
```

There is a small CUDA extension for periodic radius graphs. I install the PyG
extensions that match my PyTorch/CUDA version. For example, in my CUDA 13.3
conda environment:

```bash
MAX_JOBS=12 CUDA_HOME=/usr/local/cuda-13.3 conda run -n torch --no-capture-output \
  pip install torch-scatter --no-build-isolation --no-cache-dir
MAX_JOBS=12 CUDA_HOME=/usr/local/cuda-13.3 conda run -n torch --no-capture-output \
  pip install torch-sparse --no-build-isolation --no-cache-dir
```

The graph construction code can fall back to PyG at runtime where appropriate.
The current package build still compiles the CUDA extension, so building it
from source needs a CUDA-capable toolchain.

## Training a property model

Set the dataset path, target, and atomic graph cutoff in
[`runs/config.yaml`](runs/config.yaml), then run:

```bash
python runs/run_training.py --config runs/config.yaml
```

For a box dataset and a single target:

```bash
python runs/run_training.py --config runs/config.yaml \
  --dataset box \
  --data data/data_box_pure/example.hdf5 \
  --target HOMO \
  --radius 5.0
```

`radius` is an **atomistic** cutoff in Ångström for property training. It is
not the molecular COM cutoff used to choose surrounding context.

### Include the molecular environment

Enable context for box data to predict a query molecule in its local packing
environment. Neighbours are selected using periodic COM distances and are
included in the input graph, but only the query molecule contributes to the
supervised target and loss.

```bash
python runs/run_training.py --config runs/config.yaml \
  --dataset box \
  --data data/data_box_pure/example.hdf5 \
  --target HOMO --radius 5.0 \
  --context_mode knn --context_k 6
```

Use `--context_mode radius --context_radius 20.0` to select every molecule
within a chosen COM distance, or `--context_mode all` for the entire periodic
box (with corresponding memory cost). Context automatically uses
minimum-image edge features so interactions that cross a periodic boundary are
represented consistently.

### Use the hierarchical atomistic + COM model

The hierarchical architecture requires box context, because it needs a
molecule assignment for every atom. It alternates atomistic message passing,
atom-to-COM pooling, COM-to-COM message passing, and COM-to-atom feedback:

```bash
python runs/run_training.py --config runs/config.yaml \
  --dataset box \
  --data data/data_box_pure/example.hdf5 \
  --target HOMO --radius 5.0 \
  --context_mode radius --context_radius 20.0 \
  --model.arch hierarchical \
  --model.com_cutoff 20.0 \
  --model.num_hierarchical_layers 2
```

The atomistic radius should reflect chemical-scale interactions; `com_cutoff`
and the context cutoff should reflect the molecular packing scale. They can be
the same, but serve distinct purposes: the former defines message passing among
molecules in the hierarchy and the latter decides which molecules enter a query
sample.

## Data layout

The training runner supports two layouts:

- `molecular`: HDF5 groups containing `pos`, `types`, target property arrays,
  and optionally a `lattice`. A `(frames, atoms, 3)` `pos` array is exposed as
  one graph per frame.
- `box`: box per-molecule HDF5 files. This mode supports surrounding
  molecular context and is the required input mode for the hierarchical model.

HDF5 target keys are configurable. Supply several `--target` values (or a YAML
list) for multi-target learning; output width is inferred automatically.

## Diffusion model

The diffusion runner trains an SE(3)-equivariant denoiser for periodic
geometries. It supports atom-position generation in molecular data and
molecule-COM position generation in boxes:

```bash
python runs/run_diffusion.py --config runs/config_diffusion.yaml
```

In box diffusion mode, the radius is a **COM-scale** graph cutoff, so it should
be chosen on the scale of molecular separations rather than atomic bonds.

## Runs and evaluation

Each run writes its merged configuration, checkpoints, CSV logs, and
truth-versus-predicted plots to its own directory. W&B logging is optional;
set `logging.wandb_project` and `WANDB_API_KEY` if you use it. I keep a sample
environment file in `.env.example`.

Use `--k_folds 5` for group-aware cross-validation. For trajectories with
several molecules, frames of the same molecule remain together to avoid
train/validation leakage. Targets are standardised from training-split
statistics by default and are reported back in their original units.

More command-line and configuration examples are in
[`runs/README.md`](runs/README.md).

## Tests

```bash
pytest -q
```

## License

See [`LICENSE`](LICENSE).
