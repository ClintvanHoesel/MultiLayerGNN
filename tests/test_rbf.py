"""Tests for the RBF distance-embedding classes and their config plumbing.

Covers:
* Every built-in RBF (``GaussianRBF``, ``ExpNormalRBF``, ``BesselRBF``,
  ``ChebychevRBF``) constructs and maps distances to ``(*dist.shape, num_rbf)``
  finite features.
* ``resolve_rbf_class`` / ``RBF_REGISTRY`` turn config names into classes, keep
  instances/classes as-is, and raise a helpful error for unknown names.
* ``DistanceEmbedding`` accepts a name, class, or instance as ``rbf_class``.
* ``ScalarMoleculeModel`` builds + runs with ``rbf_kwargs.rbf_class`` set by
  name (the full config -> model path for geometric edge features).
"""

from __future__ import annotations

import inspect

import pytest
import torch

from morphology_gnn.model.embedding import DistanceEmbedding, EdgeVectorLayer
from morphology_gnn.model.rbf import (
    AbstractRBF,
    BesselRBF,
    ChebychevRBF,
    ExpNormalRBF,
    GaussianRBF,
    RBF_REGISTRY,
    resolve_rbf_class,
)
from morphology_gnn.model.scaler_model import ScalarMoleculeModel

ALL_RBF_CLASSES = [GaussianRBF, ExpNormalRBF, BesselRBF, ChebychevRBF]


# --- per-class construction + forward ----------------------------------------
@pytest.mark.parametrize("rbf_class", ALL_RBF_CLASSES)
def test_rbf_forward_shape_and_finite(rbf_class):
    m = rbf_class(cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=8)
    dist = torch.linspace(0.5, 5.0, 7)
    out = m(dist)
    assert out.shape == (7, 8)
    assert torch.isfinite(out).all()


# --- registry / resolver -----------------------------------------------------
@pytest.mark.parametrize("name", sorted(RBF_REGISTRY))
def test_resolve_rbf_class_by_name(name):
    cls = resolve_rbf_class(name)
    assert inspect.isclass(cls)
    assert issubclass(cls, AbstractRBF)


def test_resolve_rbf_class_accepts_class_and_instance():
    assert resolve_rbf_class(GaussianRBF) is GaussianRBF
    inst = ExpNormalRBF(num_rbf=4)
    assert resolve_rbf_class(inst) is inst


def test_resolve_rbf_class_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown RBF class"):
        resolve_rbf_class("NotAnRBF")


def test_exp_normal_smearing_is_alias():
    assert RBF_REGISTRY["ExpNormalSmearing"] is ExpNormalRBF


# --- DistanceEmbedding accepts names/classes/instances -----------------------
# `DistanceEmbedding` takes the RBF options as top-level kwargs (the config path
# unpacks `rbf_kwargs` with `**`), and `rbf_class` may be a name/class/instance.
@pytest.mark.parametrize("name", sorted(RBF_REGISTRY))
def test_distance_embedding_with_rbf_class_name(name):
    emb = DistanceEmbedding(num_rbf=8, rbf_class=name, cutoff_upper=5.0)
    dist = torch.linspace(0.5, 5.0, 6)
    out = emb(dist)
    assert out.shape == (6, 8)
    assert torch.isfinite(out).all()


def test_distance_embedding_with_rbf_class_instance():
    # An already-built instance is used as-is (keeps its own num_rbf).
    emb = DistanceEmbedding(num_rbf=8, rbf_class=BesselRBF(num_rbf=8))
    dist = torch.linspace(0.5, 5.0, 6)
    assert emb(dist).shape == (6, 8)


# The config path passes a nested `rbf_kwargs` dict through EdgeVectorLayer /
# ScalarDistanceEmbedding, which unpacks it into DistanceEmbedding.
@pytest.mark.parametrize(
    "name", ["GaussianRBF", "ExpNormalRBF", "BesselRBF", "ChebychevRBF"]
)
def test_edge_vector_layer_with_rbf_class_name(name):
    layer = EdgeVectorLayer(num_rbf=8, rbf_kwargs={"rbf_class": name})
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 2]]).t().contiguous()
    out = layer(pos, edge_index)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


# --- full config -> model path ------------------------------------------------
@pytest.mark.parametrize(
    "name", ["GaussianRBF", "ExpNormalRBF", "BesselRBF", "ChebychevRBF"]
)
def test_scalar_model_with_rbf_class_name(name):
    # GATConv default needs hidden_dim divisible by heads (default 3).
    model = ScalarMoleculeModel(
        hidden_dim=6,
        num_layers=1,
        use_edge_features=True,
        num_rbf=6,
        rbf_kwargs={"rbf_class": name, "cutoff_upper": 5.0},
    )
    x = torch.tensor([1, 6, 8], dtype=torch.long)  # atomic numbers (3,)
    edge_index = torch.tensor([[0, 1], [1, 2]]).t().contiguous()
    batch = torch.zeros(3, dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]])
    out = model(x, edge_index, batch, pos)
    assert out.shape == (1, 1)  # 1 graph x 1 target
    assert torch.isfinite(out).all()
