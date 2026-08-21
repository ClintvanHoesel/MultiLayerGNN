"""Tests for the envelope cutoff functions and their config plumbing.

Covers:
* Every built-in envelope (``CosineEnvelope``, ``PolynomialEnvelope``)
  constructs and returns a smooth finite mask in ``[0, 1]``.
* ``resolve_envelope`` / ``ENVELOPE_REGISTRY`` turn config names into classes,
  keep instances/classes/``None`` as-is, and raise a helpful error for unknown
  names.
* RBFs accept ``cutoff_fn`` as a name/class/instance (through the RBF
  constructor and via ``DistanceEmbedding`` / ``EdgeVectorLayer`` rbf_kwargs).
* Envelope constructor kwargs (``cutoff_fn_kwargs``, e.g. ``PolynomialEnvelope``
  ``exponent``) are passed through when the envelope is built from a name/class.
* ``ScalarMoleculeModel`` builds + runs with ``rbf_kwargs.cutoff_fn`` set by
  name (the full config -> model path for geometric edge features).
"""

from __future__ import annotations

import inspect

import pytest
import torch

from morphology_gnn.model.embedding import DistanceEmbedding, EdgeVectorLayer
from morphology_gnn.model.envelope import (
    AbstractEnvelope,
    CosineEnvelope,
    PolynomialEnvelope,
    ENVELOPE_REGISTRY,
    resolve_envelope,
)
from morphology_gnn.model.rbf import ExpNormalRBF, GaussianRBF
from morphology_gnn.model.scaler_model import ScalarMoleculeModel

ALL_ENVELOPE_CLASSES = [CosineEnvelope, PolynomialEnvelope]


# --- per-class construction + forward ----------------------------------------
@pytest.mark.parametrize("env_class", ALL_ENVELOPE_CLASSES)
def test_envelope_forward_mask(env_class):
    env = env_class(cutoff_lower=0.0, cutoff_upper=5.0)
    dist = torch.linspace(0.0, 5.0, 11)
    out = env(dist)
    assert out.shape == (11,)
    assert torch.isfinite(out).all()
    assert (out >= 0).all() and (out <= 1).all()


# --- registry / resolver -----------------------------------------------------
@pytest.mark.parametrize("name", sorted(ENVELOPE_REGISTRY))
def test_resolve_envelope_by_name(name):
    cls = resolve_envelope(name)
    assert inspect.isclass(cls)
    assert issubclass(cls, AbstractEnvelope)


def test_resolve_envelope_accepts_class_instance_none():
    assert resolve_envelope(CosineEnvelope) is CosineEnvelope
    inst = PolynomialEnvelope()
    assert resolve_envelope(inst) is inst
    assert resolve_envelope(None) is None


def test_resolve_envelope_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown cutoff_fn"):
        resolve_envelope("NotAnEnvelope")


# PyYAML parses bare `None` as the string "None" (only `null` -> Python None);
# both spellings must disable the cutoff rather than crash the trial.
@pytest.mark.parametrize("no_cutoff", [None, "None", "null", "none", " NULL "])
def test_resolve_envelope_disables_cutoff(no_cutoff):
    assert resolve_envelope(no_cutoff) is None


def test_rbf_with_string_none_cutoff_disables_envelope():
    rbf = GaussianRBF(cutoff_upper=5.0, num_rbf=4, cutoff_fn="None")
    assert rbf.cutoff_fn is None


# --- RBFs accept cutoff_fn by name -------------------------------------------
@pytest.mark.parametrize("name", sorted(ENVELOPE_REGISTRY))
def test_rbf_with_cutoff_fn_name(name):
    rbf = GaussianRBF(cutoff_upper=5.0, num_rbf=4, cutoff_fn=name)
    dist = torch.linspace(0.5, 5.0, 6)
    out = rbf(dist)
    assert out.shape == (6, 4)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", sorted(ENVELOPE_REGISTRY))
def test_distance_embedding_with_cutoff_fn_name(name):
    emb = DistanceEmbedding(
        num_rbf=4, rbf_class=ExpNormalRBF, cutoff_fn=name, cutoff_upper=5.0
    )
    dist = torch.linspace(0.5, 5.0, 6)
    out = emb(dist)
    assert out.shape == (6, 4)
    assert torch.isfinite(out).all()


def test_edge_vector_layer_with_cutoff_fn_name():
    layer = EdgeVectorLayer(num_rbf=4, rbf_kwargs={"cutoff_fn": "PolynomialEnvelope"})
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 2]]).t().contiguous()
    out = layer(pos, edge_index)
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()


# --- full config -> model path ------------------------------------------------
@pytest.mark.parametrize("name", ["CosineEnvelope", "PolynomialEnvelope"])
def test_scalar_model_with_cutoff_fn_name(name):
    # GATConv default needs hidden_dim divisible by heads (default 3).
    model = ScalarMoleculeModel(
        hidden_dim=6,
        num_layers=1,
        use_edge_features=True,
        num_rbf=6,
        rbf_kwargs={
            "rbf_class": "BesselRBF",
            "cutoff_fn": name,
            "cutoff_upper": 5.0,
        },
    )
    x = torch.tensor([1, 6, 8], dtype=torch.long)  # atomic numbers (3,)
    edge_index = torch.tensor([[0, 1], [1, 2]]).t().contiguous()
    batch = torch.zeros(3, dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]])
    out = model(x, edge_index, batch, pos)
    assert out.shape == (1, 1)  # 1 graph x 1 target
    assert torch.isfinite(out).all()


# --- cutoff_fn_kwargs: envelope constructor parameters -----------------------
def test_rbf_with_cutoff_fn_class_and_kwargs():
    rbf = GaussianRBF(
        cutoff_upper=5.0,
        num_rbf=4,
        cutoff_fn=PolynomialEnvelope,
        cutoff_fn_kwargs={"exponent": 7},
    )
    assert isinstance(rbf.cutoff_fn, PolynomialEnvelope)
    assert rbf.cutoff_fn.exponent == 7


def test_rbf_with_cutoff_fn_name_and_kwargs():
    rbf = ExpNormalRBF(
        cutoff_upper=5.0,
        num_rbf=4,
        cutoff_fn="PolynomialEnvelope",
        cutoff_fn_kwargs={"exponent": 3},
    )
    assert isinstance(rbf.cutoff_fn, PolynomialEnvelope)
    assert rbf.cutoff_fn.exponent == 3


def test_distance_embedding_with_cutoff_fn_kwargs():
    emb = DistanceEmbedding(
        num_rbf=4,
        rbf_class=ExpNormalRBF,
        cutoff_fn="PolynomialEnvelope",
        cutoff_fn_kwargs={"exponent": 7},
        cutoff_upper=5.0,
    )
    assert isinstance(emb.rbf.cutoff_fn, PolynomialEnvelope)
    assert emb.rbf.cutoff_fn.exponent == 7
    dist = torch.linspace(0.5, 5.0, 6)
    assert torch.isfinite(emb(dist)).all()


def test_edge_vector_layer_with_cutoff_fn_kwargs():
    layer = EdgeVectorLayer(
        num_rbf=4,
        rbf_kwargs={
            "cutoff_fn": "PolynomialEnvelope",
            "cutoff_fn_kwargs": {"exponent": 9},
        },
    )
    assert isinstance(layer.edge_emb.rbf_emb.rbf.cutoff_fn, PolynomialEnvelope)
    assert layer.edge_emb.rbf_emb.rbf.cutoff_fn.exponent == 9
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 2]]).t().contiguous()
    assert torch.isfinite(layer(pos, edge_index)).all()


def test_scalar_model_with_cutoff_fn_kwargs():
    model = ScalarMoleculeModel(
        hidden_dim=6,
        num_layers=1,
        use_edge_features=True,
        num_rbf=6,
        rbf_kwargs={
            "rbf_class": "BesselRBF",
            "cutoff_fn": "PolynomialEnvelope",
            "cutoff_fn_kwargs": {"exponent": 5},
            "cutoff_upper": 5.0,
        },
    )
    assert isinstance(
        model.edge_layer.edge_emb.rbf_emb.rbf.cutoff_fn, PolynomialEnvelope
    )
    assert model.edge_layer.edge_emb.rbf_emb.rbf.cutoff_fn.exponent == 5
    x = torch.tensor([1, 6, 8], dtype=torch.long)
    edge_index = torch.tensor([[0, 1], [1, 2]]).t().contiguous()
    batch = torch.zeros(3, dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]])
    out = model(x, edge_index, batch, pos)
    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()
