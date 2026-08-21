import inspect
import logging
from abc import ABC, abstractmethod
import math

import torch

from .envelope import AbstractEnvelope, CosineEnvelope

logger = logging.getLogger(__name__)


class AbstractRBF(torch.nn.Module, ABC):
    """Abstract base class for radial basis function (RBF) distance embeddings.

    Use this as the typing anchor for any RBF module — e.g.
    ``rbf_class: type[AbstractRBF]`` or ``rbf: AbstractRBF`` — and for
    ``isinstance`` checks over concrete implementations such as
    :class:`GaussianRBF` and :class:`ExpNormalSmearing`.

    Subclasses must implement :meth:`reset_parameters` and :meth:`forward`;
    the shared ``(cutoff_lower, cutoff_upper, num_rbf, trainable, dtype)``
    constructor arguments and the optional ``cutoff_fn`` are stored on the
    base.

    ``cutoff_fn`` is an optional smooth cutoff envelope multiplied into the
    RBF features — an :class:`AbstractEnvelope` instance, an
    ``AbstractEnvelope`` subclass (built with this RBF's cutoffs), or
    ``None`` for no cutoff (effectively multiplying by ``1``).
    """

    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = False,
        dtype: torch.dtype = torch.float32,
        cutoff_fn: AbstractEnvelope | type[AbstractEnvelope] | None = None,
    ) -> None:
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.dtype = dtype
        self.cutoff_fn = self._build_cutoff_fn(cutoff_fn)

    def _build_cutoff_fn(
        self, cutoff_fn: AbstractEnvelope | type[AbstractEnvelope] | None
    ) -> AbstractEnvelope | None:
        """Resolve ``cutoff_fn`` (instance, class, or ``None``) to an envelope."""
        if cutoff_fn is None:
            return None
        if isinstance(cutoff_fn, AbstractEnvelope):
            return cutoff_fn
        if inspect.isclass(cutoff_fn) and issubclass(cutoff_fn, AbstractEnvelope):
            return cutoff_fn(
                cutoff_lower=self.cutoff_lower, cutoff_upper=self.cutoff_upper
            )
        raise TypeError(
            "cutoff_fn must be an AbstractEnvelope instance, an "
            f"AbstractEnvelope subclass, or None; got {cutoff_fn!r}"
        )

    def _apply_cutoff(self, dist: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Multiply ``features`` by the cutoff envelope (no-op if ``None``)."""
        if self.cutoff_fn is None:
            return features
        # ``features`` has shape ``(*dist.shape, num_rbf)``; align the envelope
        # output (which follows ``dist.shape``) to the leading dims so it
        # broadcasts against the trailing ``num_rbf`` feature dimension.
        cutoff = self.cutoff_fn(dist)
        return features * cutoff.reshape(features.shape[:-1] + (1,))

    @abstractmethod
    def reset_parameters(self) -> None:
        """(Re)initialize the RBF basis parameters."""

    @abstractmethod
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Map distances to RBF features of shape ``(*dist.shape, num_rbf)``."""


# https://github.com/torchmd/torchmd-net/blob/main/torchmdnet/models/utils.py
class GaussianRBF(AbstractRBF):
    offset: torch.Tensor
    coeff: torch.Tensor

    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = False,
        dtype: torch.dtype = torch.float32,
        cutoff_fn: AbstractEnvelope | type[AbstractEnvelope] | None = None,
    ):
        super().__init__(
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            num_rbf=num_rbf,
            trainable=trainable,
            dtype=dtype,
            cutoff_fn=cutoff_fn,
        )
        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", torch.nn.Parameter(coeff))
            self.register_parameter("offset", torch.nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(
            self.cutoff_lower, self.cutoff_upper, self.num_rbf, dtype=self.dtype
        )
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self) -> None:
        offset, coeff = self._initial_params()
        with torch.no_grad():
            self.offset.copy_(offset)
            self.coeff.copy_(coeff)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        features = torch.exp(
            self.coeff * torch.pow(dist.unsqueeze(-1) - self.offset, 2)
        )
        return self._apply_cutoff(dist, features)


class ExpNormalRBF(AbstractRBF):
    means: torch.Tensor
    betas: torch.Tensor

    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = False,
        dtype: torch.dtype = torch.float32,
        cutoff_fn: AbstractEnvelope | type[AbstractEnvelope] | None = CosineEnvelope,
    ):
        super().__init__(
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            num_rbf=num_rbf,
            trainable=trainable,
            dtype=dtype,
            cutoff_fn=cutoff_fn,
        )

        means, betas, alpha = self._initial_params()
        self.alpha = alpha
        if trainable:
            self.register_parameter("means", torch.nn.Parameter(means))
            self.register_parameter("betas", torch.nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(
                -self.cutoff_upper + self.cutoff_lower, dtype=self.dtype
            )
        )
        means = torch.linspace(start_value, 1, self.num_rbf, dtype=self.dtype)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf,
            dtype=self.dtype,
        )
        alpha = 5.0 / (self.cutoff_upper - self.cutoff_lower)
        return means, betas, alpha

    def reset_parameters(self) -> None:
        means, betas, alpha = self._initial_params()
        self.alpha = alpha
        with torch.no_grad():
            self.means.copy_(means)
            self.betas.copy_(betas)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1)
        features = torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )
        return self._apply_cutoff(dist, features)


class BesselRBF(AbstractRBF):
    """
    Adapted from https://github.com/ACEsuit/mace/blob/main/mace/modules/radial.py
    Equation (7)
    """

    bessel_weights: torch.Tensor

    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = False,
        dtype: torch.dtype = torch.float32,
        cutoff_fn: AbstractEnvelope | type[AbstractEnvelope] | None = CosineEnvelope,
    ):
        super().__init__(
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            num_rbf=num_rbf,
            trainable=trainable,
            dtype=dtype,
            cutoff_fn=cutoff_fn,
        )

        bessel_weights, prefactor = self._initial_params()
        self.prefactor = prefactor

        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def _initial_params(self):
        bessel_weights = (
            torch.pi
            / (self.cutoff_upper - self.cutoff_lower)
            * torch.linspace(
                start=1.0,
                end=self.num_rbf,
                steps=self.num_rbf,
                dtype=self.dtype,
            )
        )
        prefactor = math.sqrt(2.0 / (self.cutoff_upper - self.cutoff_lower))
        return bessel_weights, prefactor

    def reset_parameters(self) -> None:
        bessel_weights, prefactor = self._initial_params()
        self.prefactor = prefactor
        with torch.no_grad():
            self.bessel_weights.copy_(bessel_weights)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # [..., 1] so it broadcasts against bessel_weights of shape (num_rbf,).
        ddist = (dist - self.cutoff_lower).unsqueeze(-1)
        numerator = torch.sin(self.bessel_weights * ddist)  # [..., num_rbf]
        # Regularize the denominator to avoid 0/0 -> NaN at the cutoff_lower.
        eps = torch.finfo(self.dtype).eps
        features = self.prefactor * (numerator / (ddist + eps))
        return self._apply_cutoff(dist, features)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(cutoff=({self.cutoff_lower}, {self.cutoff_upper}), "
            f"num_rbf={self.num_rbf}, trainable={self.bessel_weights.requires_grad})"
        )


class ChebychevRBF(AbstractRBF):
    """Chebyshev polynomial (first kind) radial basis on ``[cutoff_lower, cutoff_upper]``.

    Distances are rescaled from ``[cutoff_lower, cutoff_upper]`` onto the
    Chebyshev domain ``[-1, 1]`` and expanded in the Chebyshev polynomials
    ``T_n`` for ``n = 1..num_rbf``, then gated by the ``cutoff_fn`` envelope.
    The polynomial orders are fixed integer buffers, so ``trainable`` is
    accepted for interface compatibility with :class:`DistanceEmbedding` but
    has no effect on this basis.

    Reference: adapted from MACE ``mace/modules/radial.py``.
    """

    n: torch.Tensor

    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 8,
        trainable: bool = False,
        dtype: torch.dtype = torch.float32,
        cutoff_fn: AbstractEnvelope | type[AbstractEnvelope] | None = CosineEnvelope,
    ) -> None:
        super().__init__(
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            num_rbf=num_rbf,
            trainable=trainable,
            dtype=dtype,
            cutoff_fn=cutoff_fn,
        )
        # Orders are non-trainable integer indices into the polynomial family.
        n, prefactor = self._initial_params()
        self.prefactor = prefactor
        self.register_buffer("n", n)

    def _initial_params(self):
        n = torch.arange(1, self.num_rbf + 1, dtype=torch.long)
        prefactor = 2.0 / (self.cutoff_upper - self.cutoff_lower)
        return n, prefactor

    def reset_parameters(self) -> None:
        n, prefactor = self._initial_params()
        self.prefactor = prefactor
        with torch.no_grad():
            self.n.copy_(n)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # Map [cutoff_lower, cutoff_upper] onto [-1, 1], the domain of T_n.
        x = self.prefactor * (dist - self.cutoff_lower) - 1.0
        features = torch.special.chebyshev_polynomial_t(x.unsqueeze(-1), self.n)
        return self._apply_cutoff(dist, features)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(cutoff=({self.cutoff_lower}, {self.cutoff_upper}), "
            f"num_rbf={self.num_rbf}, trainable={self.trainable})"
        )
