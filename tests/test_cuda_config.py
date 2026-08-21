"""Tests for the ``cuda.tensor_cores`` config knob (:func:`configure_cuda`).

Covers:
* With ``cuda.tensor_cores: false`` (default) nothing changes: the torch
  float32 matmul precision is left as-is.
* With ``cuda.tensor_cores: true``: precision is set to ``"high"`` (TF32 /
  Tensor Cores), which also prevents the PyTorch "Tensor Cores" nag from
  being emitted (it only fires at the default "highest" precision).
* A missing/empty ``cuda`` section behaves like ``false``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# `configure_cuda` lives in the runner helpers (`runs/training_helpers.py`), so
# put `runs/` on the path (the run scripts do the same).
_RUNS = Path(__file__).resolve().parents[1] / "runs"
if str(_RUNS) not in sys.path:
    sys.path.insert(0, str(_RUNS))

from training_helpers import configure_cuda  # noqa: E402


@pytest.fixture
def _torch_globals():
    """Snapshot and restore the global torch matmul precision."""
    precision = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(precision)


def test_tensor_cores_disabled_leaves_defaults(_torch_globals):
    configure_cuda({"cuda": {"tensor_cores": False}})
    assert torch.get_float32_matmul_precision() != "high"


def test_tensor_cores_missing_section_is_noop(_torch_globals):
    configure_cuda({})
    assert torch.get_float32_matmul_precision() != "high"


def test_tensor_cores_enabled_sets_precision(_torch_globals):
    configure_cuda({"cuda": {"tensor_cores": True}})
    assert torch.get_float32_matmul_precision() == "high"
