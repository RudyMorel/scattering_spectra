from typing import *
import numpy as np
import torch.nn as nn
import torch

import utils.complex_utils as cplx
from scattering_network.described_tensor import DescribedTensor


class MSELossScat(nn.Module):
    def __init__(self):
        super(MSELossScat, self).__init__()
        self.max_gap = {}
        self.lam_cutoff = 1e-5

    def compute_gap(self, input: Optional[DescribedTensor], target: DescribedTensor, weights):
        if input is None:
            gap = torch.zeros_like(target.y) - target.y
        else:
            gap = input.y - target.y

        gap = gap if weights is None else weights.unsqueeze(-1) * gap
        self.max_gap = {m_type: torch.max(cplx.modulus(gap[target.idx_info.where(m_type=m_type)])).item()
                        for m_type in np.unique(target.idx_info['m_type'])}
        return gap

    def forward(self, input, target, weights_gap, weights_l2):
        gap = self.compute_gap(input, target, weights_gap)
        if weights_l2 is None:
            return cplx.modulus(gap).pow(2.0).mean()
        return (weights_l2 * cplx.modulus(gap).pow(2.0)).sum()


class MSELossCov(nn.Module):
    def __init__(self):
        super(MSELossCov, self).__init__()
        self.max_gap = {}

    def forward(self, input, target):
        gap = torch.zeros_like(target.y)

        # exp
        gap[target.idx_info.where(q=1), :] = cplx.mul(target.select(q=1), input.select(q=1) - target.select(q=1))

        # cov
        gap[target.idx_info.where(q=2), :] = input.select(q=2) - target.select(q=2)

        self.max_gap = {m_type: torch.max(cplx.modulus(gap[target.idx_info.where(m_type=m_type)])).item()
                        for m_type in np.unique(target.idx_info['m_type'])}

        return cplx.modulus(gap).pow(2.0).mean()
