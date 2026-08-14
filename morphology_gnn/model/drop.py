import logging

import torch

logger = logging.getLogger(__name__)


# https://github.com/atomicarchitects/equiformer_v3/blob/main/experimental/models/equiformer_v3/drop.py
class EquivariantDropout(torch.nn.Module):
    """
    1.  When dropping one type-L vector, we set all the m components to zeros.
    """

    def __init__(self, lmax, mmax, drop_prob, use_m_primary=False):
        super(EquivariantDropout, self).__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.drop_prob = drop_prob
        self.use_m_primary = use_m_primary
        logger.debug(
            "EquivariantDropout(lmax=%d, mmax=%d, drop_prob=%s, use_m_primary=%s)",
            lmax,
            mmax,
            drop_prob,
            use_m_primary,
        )

        self.drop = torch.nn.Dropout(drop_prob, True)

        expand_index = []
        if not self.use_m_primary:
            for l in range(self.lmax + 1):
                mmax = min(l, self.mmax)
                l_index_tensor = torch.ones(((2 * mmax + 1),), dtype=torch.long) * l
                expand_index.append(l_index_tensor)
        elif self.use_m_primary:
            for m in range(self.mmax + 1):
                l_index = torch.arange((self.lmax + 1 - m))
                expand_index.append(l_index)
                if m > 0:
                    expand_index.append(l_index)  # +- m
        expand_index = torch.cat(expand_index, dim=0)
        expand_index = expand_index.long()
        self.register_buffer("expand_index", expand_index)

    def extra_repr(self):
        return "lmax={}, mmax={}, drop_prob={}, use_m_primary={}".format(
            self.lmax, self.mmax, self.drop_prob, self.use_m_primary
        )

    def forward(self, x):
        # x shape: (num_tokens, num_m_coefficients, num_channels)
        if not self.training or self.drop_prob == 0.0:
            return x

        assert len(x.shape) == 3
        shape = (x.shape[0], (self.lmax + 1), x.shape[2])
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        mask = torch.index_select(mask, dim=1, index=self.expand_index)
        out = x * mask
        return out
