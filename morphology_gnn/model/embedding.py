import inspect
import logging

import torch

from .rbf import GaussianRBF

logger = logging.getLogger(__name__)

# Largest atomic number in the standard periodic table (Oganesson, Z=118).
MAX_ATOMIC_NUMBER = 118


class AtomTypeEmbedding(torch.nn.Module):
    """Embed atomic numbers (Z) into a dense vector space."""

    def __init__(
        self, embedding_dim: int, max_atomic_number: int = MAX_ATOMIC_NUMBER, **kwargs
    ) -> None:
        super().__init__()
        # +1 so that atomic number Z maps directly to row Z (indexes are 0-based).
        self.num_embeddings = max_atomic_number + 1
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(
            self.num_embeddings,
            embedding_dim,
            **kwargs,
        )

    def forward(self, atom_types: torch.Tensor) -> torch.Tensor:
        """Map a tensor of atomic numbers to embeddings.

        Args:
            atom_types: Atomic numbers, shape ``(N,)`` (or ``(N, 1)``, as stored
                by PyG datasets). Converted to ``torch.long`` if necessary.

        Returns:
            Embeddings of shape ``(N, embedding_dim)``.
        """
        if atom_types.dtype != torch.long:
            atom_types = atom_types.long()
        # PyG datasets commonly store atom types as (N, 1); drop the singleton so
        # nn.Embedding does not return a 3-D (N, 1, embedding_dim) tensor.
        if atom_types.dim() > 1 and atom_types.shape[-1] == 1:
            atom_types = atom_types.squeeze(-1)
        return self.embedding(atom_types)


class DistanceEmbedding(torch.nn.Module):
    """Embed inter-atomic distance vectors into a dense feature space.

    The output size (``num_rbf``) is configurable and handed to an arbitrary RBF
    class — ``GaussianRBF``, ``ExpNormalSmearing``, or any custom RBF with a
    matching ``(cutoff_lower, cutoff_upper, num_rbf, ...)`` constructor. This
    mirrors :class:`AtomTypeEmbedding` (which maps discrete atom types) while
    mapping continuous distances through a radial basis that can optionally be
    trained end-to-end (``trainable=True``).
    """

    def __init__(
        self,
        num_rbf: int = 50,
        rbf_class=GaussianRBF,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        trainable: bool = True,
        dtype: torch.dtype = torch.float32,
        **rbf_kwargs,
    ) -> None:
        super().__init__()
        self.num_rbf = num_rbf
        self.embedding_dim = num_rbf
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        # `rbf_class` may be a module *class* or an already-built RBF instance.
        if inspect.isclass(rbf_class):
            self.rbf = rbf_class(
                cutoff_lower=cutoff_lower,
                cutoff_upper=cutoff_upper,
                num_rbf=num_rbf,
                trainable=trainable,
                dtype=dtype,
                **rbf_kwargs,
            )
        else:
            self.rbf = rbf_class

    def reset_parameters(self) -> None:
        if hasattr(self.rbf, "reset_parameters"):
            self.rbf.reset_parameters()

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Map a tensor of distances to RBF features.

        Args:
            distances: Pairwise distances, any shape (e.g. ``(E,)`` for a flat
                edge list or ``(E, 1)``).

        Returns:
            Features of shape ``(*distances.shape, num_rbf)``.
        """
        return self.rbf(distances)


class DistanceVectors(torch.nn.Module):
    """Decompose displacement (distance) vectors into unit vectors and distances.

    Given relative position vectors between pairs of atoms, e.g.
    ``r_ij = x_j - x_i`` of shape ``(..., 3)``, this computes the *normalized*
    direction vectors ``r_hat = r / ||r||`` and the scalar distances
    ``d = ||r||`` in a single pass::

        unit, dist = DistanceVectors()(r_ij)

    Zero-length vectors (e.g. self-loops where both endpoints coincide) are
    handled gracefully: their distance is ``0`` and their direction is the zero
    vector, so no ``NaN`` is produced. The direction is computed via a mask, so
    zero-length vectors also get a *zero gradient* rather than the ``1/eps``
    blow-up that naive ``v / clamp_min(norm)`` produces.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize a batch of displacement vectors.

        Args:
            vectors: Displacement vectors, shape ``(..., D)`` (typically ``D=3``).

        Returns:
            A tuple ``(unit_vectors, distances)``:

            - ``unit_vectors``: shape ``(..., D)`` — unit norm where the input is
              nonzero, and the zero vector where the input is zero.
            - ``distances``: shape ``(...,)`` — the true Euclidean norm.
        """
        distances = vectors.norm(dim=-1)  # (...,)
        nonzero = distances > self.eps  # (...,)
        unit_vectors = torch.where(
            nonzero.unsqueeze(-1),
            vectors / distances.clamp_min(self.eps).unsqueeze(-1),
            torch.zeros_like(vectors),
        )
        return unit_vectors, distances


class ScalarDistanceEmbedding(torch.nn.Module):
    """Embed edge displacement vectors into scalar edge features.

    Wraps :class:`DistanceVectors` (normalized direction + distances) and
    :class:`DistanceEmbedding` (RBF-embedded distances) to turn raw displacement
    vectors of shape ``(E, 3)`` into edge features of shape ``(E, num_rbf)``.

    ``rbf_kwargs`` override the first-class defaults when both are given::

        emb = ScalarDistanceEmbedding()                       # all defaults
        emb = ScalarDistanceEmbedding(num_rbf=64)
        emb = ScalarDistanceEmbedding(
            rbf_kwargs={"cutoff_upper": 6.0, "rbf_class": ExpNormalSmearing}
        )
    """

    def __init__(
        self,
        num_rbf: int = 50,
        rbf_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        # `None` instead of a mutable default; copy so we never mutate caller dicts.
        rbf_kwargs = dict(rbf_kwargs or {})
        # kwargs win over the first-class defaults when both are supplied.
        rbf_kwargs.setdefault("num_rbf", num_rbf)

        self.distance_emb = DistanceVectors()
        self.rbf_emb = DistanceEmbedding(**rbf_kwargs)

    @property
    def embedding_dim(self) -> int:
        return self.rbf_emb.embedding_dim

    def forward(self, displacement_vectors: torch.Tensor) -> torch.Tensor:
        """Embed a batch of edge displacement vectors.

        Args:
            displacement_vectors: Relative position vectors for all edges,
                shape ``(E, 3)``.

        Returns:
            Edge features of shape ``(E, num_rbf)`` — RBF embedding of the edge
            distances.
        """
        _, distances = self.distance_emb(displacement_vectors)
        return self.rbf_emb(distances)


class EdgeVectorLayer(torch.nn.Module):
    """Compute edge attributes from node positions.

    For every edge ``(src, dst)`` this builds the displacement vector
    ``pos[dst] - pos[src]`` and embeds it with :class:`ScalarDistanceEmbedding`
    (unit vector + RBF-embedded distance) into an edge attribute of width
    ``num_rbf`` — ready for message-passing layers that accept ``edge_attr``
    (e.g. ``GATConv`` with ``edge_dim=num_rbf``)::

        edge_attr = EdgeVectorLayer(num_rbf=16)(pos, edge_index)  # (E, 16)
        x = conv(x, edge_index, edge_attr=edge_attr)
    """

    def __init__(self, num_rbf: int = 50, rbf_kwargs: dict | None = None) -> None:
        super().__init__()
        self.edge_emb = ScalarDistanceEmbedding(num_rbf=num_rbf, rbf_kwargs=rbf_kwargs)

    @property
    def embedding_dim(self) -> int:
        return self.edge_emb.embedding_dim

    def forward(self, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Embed edge displacements from node positions.

        Args:
            pos: Node positions, shape ``(N, 3)``.
            edge_index: Connectivity, shape ``(2, E)``.

        Returns:
            Edge attributes of shape ``(E, num_rbf)``.
        """
        logger.debug(
            "EdgeVectorLayer: %d edges, %d nodes, num_rbf=%d",
            edge_index.shape[1],
            pos.shape[0],
            self.embedding_dim,
        )
        displacement = pos[edge_index[1]] - pos[edge_index[0]]  # (E, 3)
        return self.edge_emb(displacement)
