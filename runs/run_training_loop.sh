#!/usr/bin/env bash
#
# Loop `runs/run_training.py` over every HDF5 file in a folder.
#
# For each *.hdf5 (or *.h5) file in DATA_DIR it runs:
#
#   python runs/run_training.py --config <config> --data <file> [extra args...]
#
# and isolates the output directory per file (outdir = runs/artifacts/<stem>)
# so checkpoints / truth-vs-predicted plots never overwrite each other.
#
# Usage examples:
#   ./runs/run_training_loop.sh                                        # defaults
#   ./runs/run_training_loop.sh --data-dir data/data_box_pure --radius 20.0
#   ./runs/run_training_loop.sh --python $HOME/miniforge3/envs/torch/bin/python \
#       --max_epochs 100 --wandb_project MyProject
#   ./runs/run_training_loop.sh --dry-run --data-dir data/data_Daniel   # just print
#
# Script options (everything else is passed through to run_training.py):
#   --data-dir DIR         Folder with HDF5 files (default: data/data_Daniel)
#   --config FILE          Config file (default: runs/config.yaml)
#   --python BIN           Python interpreter (default: $PYTHON, else 'python')
#   --recursive            Also recurse into subfolders
#   --stop-on-error        Abort on the first failed run (default: continue)
#   --no-outdir-per-file   Don't isolate the outdir per file
#   --dry-run              Print the commands without running them
#   -h, --help             Show this help
#
# Common passthroughs: --radius --target --max_epochs --batch_size --lr
#   --outdir --wandb_project --group --model.rbf_kwargs.rbf_class ...
#   (a user-supplied --outdir is applied to every file; the per-file isolation
#    is then disabled automatically)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${DATA_DIR:-$ROOT/data/data_Daniel}"
CONFIG="${CONFIG:-$ROOT/runs/config.yaml}"
PYTHON_BIN="${PYTHON:-python}"
RECURSIVE=0
STOP_ON_ERROR=0
OUTDIR_PER_FILE=1
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [script options] [run_training args...]

Loops runs/run_training.py over every HDF5 file in a folder.

Script options:
  --data-dir DIR          Folder with .hdf5/.h5 files
                          (default: $ROOT/data/data_Daniel)
  --config FILE           Config file (default: $ROOT/runs/config.yaml)
  --python BIN            Python interpreter (default: \$PYTHON or 'python')
  --recursive             Recurse into subfolders
  --stop-on-error         Abort on the first failed run (default: continue)
  --no-outdir-per-file    Don't isolate the outdir per file
  --dry-run               Print the commands without running them
  -h, --help              Show this help

Anything else is passed through to run_training.py, e.g.:
  --radius 4.5 --target HOMO --max_epochs 100 --wandb_project MyProject
  --model.rbf_kwargs.rbf_class ExpNormalRBF
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --recursive) RECURSIVE=1; shift ;;
    --stop-on-error) STOP_ON_ERROR=1; shift ;;
    --no-outdir-per-file) OUTDIR_PER_FILE=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: data dir not found: $DATA_DIR" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found: $CONFIG" >&2
  exit 1
fi

if [[ $RECURSIVE -eq 1 ]]; then
  mapfile -t FILES < <(find "$DATA_DIR" -type f \( -name '*.hdf5' -o -name '*.h5' \) | sort)
else
  mapfile -t FILES < <(find "$DATA_DIR" -maxdepth 1 -type f \( -name '*.hdf5' -o -name '*.h5' \) | sort)
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "ERROR: no HDF5 files found in $DATA_DIR" >&2
  exit 1
fi

echo "Training on ${#FILES[@]} HDF5 file(s) from: $DATA_DIR"
echo "Python: $PYTHON_BIN"
echo "Config: $CONFIG"
echo "Pass-through args: ${EXTRA_ARGS[*]:-<none>}"
echo

FAILED=()
for file in "${FILES[@]}"; do
  base="$(basename "$file")"
  stem="${base%.*}"

  # Detect a user-supplied --outdir (applies to every file -> disable isolation).
  has_outdir=0
  for a in "${EXTRA_ARGS[@]}"; do
    [[ "$a" == "--outdir" ]] && has_outdir=1
  done

  args=(--config "$CONFIG" --data "$file")
  if [[ $OUTDIR_PER_FILE -eq 1 && $has_outdir -eq 0 ]]; then
    args+=(--outdir "$ROOT/runs/artifacts/$stem")
  fi
  args+=("${EXTRA_ARGS[@]}")

  echo "======================================================================"
  echo ">>> Training on: $file"
  echo ">>> Command: $PYTHON_BIN $ROOT/runs/run_training.py ${args[*]}"
  echo "======================================================================"

  if [[ $DRY_RUN -eq 1 ]]; then
    continue
  fi

  if ! "$PYTHON_BIN" "$ROOT/runs/run_training.py" "${args[@]}"; then
    echo "ERROR: run failed for $base (exit code $?)" >&2
    FAILED+=("$base")
    if [[ $STOP_ON_ERROR -eq 1 ]]; then
      echo "Aborting (--stop-on-error)." >&2
      exit 1
    fi
  fi
done

echo
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "Done: all ${#FILES[@]} run(s) finished successfully."
else
  echo "Done with ${#FAILED[@]} failure(s): ${FAILED[*]}" >&2
  exit 1
fi
