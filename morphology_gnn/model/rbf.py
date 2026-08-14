import logging
from abc import ABC, abstractmethod

import torch

from .envelope import CosineEnvelope

logger = logging.getLogger(__name__)


class AbstractRBF(torch.nn.Module, ABC):
    """Abstract base class for radial basis function (RBF) distance embeddings.

    Use this as the typing anchor for any RBF module — e.g.
    ``rbf_class: type[AbstractRBF]`` or ``rbf: AbstractRBF`` — and for
    ``isinstance`` checks over concrete implementations such as
    :class:`GaussianRBF` and :class:`ExpNormalSmearing`.

    Subclasses must implement :meth:`reset_parameters` and :meth:`forward`;
    the shared ``(cutoff_lower, cutoff_upper, num_rbf, trainable, dtype)``
    constructor arguments are stored on the base.
    """

    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.dtype = dtype

    @abstractmethod
    def reset_parameters(self) -> None:
        """(Re)initialize the RBF basis parameters."""

    @abstractmethod
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Map distances to RBF features of shape ``(*dist.shape, num_rbf)``."""


# https://github.com/torchmd/torchmd-net/blob/main/torchmdnet/models/utils.py
class GaussianRBF(AbstractRBF):
    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            num_rbf=num_rbf,
            trainable=trainable,
            dtype=dtype,
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
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalSmearing(AbstractRBF):

    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            num_rbf=num_rbf,
            trainable=trainable,
            dtype=dtype,
        )
        self.cutoff_fn = CosineEnvelope(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
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
        return means, betas

    def reset_parameters(self) -> None:
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )
