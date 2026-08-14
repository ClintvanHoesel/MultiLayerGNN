import logging
import math

import torch

logger = logging.getLogger(__name__)


# https://github.com/atomicarchitects/equiformer_v3/blob/main/experimental/models/equiformer_v3/envelope.py
# https://github.com/gasteigerjo/dimenet/blob/master/dimenet/model/layers/envelope.py
class PolynomialEnvelope(torch.nn.Module):
    """
    1.  Polynomial envelope function that ensures a smooth cutoff.
    2.  Reference: https://github.com/facebookresearch/fairchem/blob/518d0ea12110548bd5ffaf9a43060b8eae152e13/src/fairchem/core/models/esen/nn/radial.py#L22
    """

    def __init__(
        self, cutoff_lower: float = 0.0, cutoff_upper: float = 5.0, exponent: int = 5
    ) -> None:
        super().__init__()
        assert exponent > 0
        self.cutoff_lower = float(cutoff_lower)
        self.cutoff_upper = float(cutoff_upper)
        self.cutoff_diff = self.cutoff_upper - self.cutoff_lower
        self.exponent = exponent
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d_scaled = (distances - self.cutoff_lower) / self.cutoff_diff
        cutoffs = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        # cutoffs = torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))
        # outputs = outputs.view(-1, 1)
        cutoffs = cutoffs * (distances < self.cutoff_upper)
        cutoffs = cutoffs * (distances > self.cutoff_lower)
        return cutoffs

    def extra_repr(self):
        return "cutoff={}, exponent={}".format(self.cutoff, self.exponent)


class CosineEnvelope(torch.nn.Module):

    def __init__(self, cutoff_lower: float = 0.0, cutoff_upper: float = 5.0):
        super(CosineEnvelope, self).__init__()
        self.cutoff_lower = float(cutoff_lower)
        self.cutoff_upper = float(cutoff_upper)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            cutoffs = cutoffs * (distances > self.cutoff_lower)
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            return cutoffs
