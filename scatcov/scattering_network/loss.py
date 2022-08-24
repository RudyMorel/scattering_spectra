""" Loss functions on scattering moments. """
from typing import *
import numpy as np
import torch.nn as nn
import torch

import scatcov.utils.complex_utils as cplx
from scatcov.scattering_network.described_tensor import DescribedTensor


class MSELossScat(nn.Module):
    """ Implements l2 norm on the scattering coefficients or scattering covariances. """
    def __init__(self):
        super(MSELossScat, self).__init__()
        self.max_gap = {}

    def compute_gap(self, input: Optional[DescribedTensor], target: DescribedTensor, weights):
        if input is None:
            gap = torch.zeros_like(target.y) - target.y
        else:
            gap = input.y - target.y
        gap = gap[0, :, 0, :]

        gap = gap if weights is None else weights.unsqueeze(-1) * gap
        self.max_gap = {m_type: torch.max(cplx.modulus(gap[target.descri.where(m_type=m_type), :])).item()
                        for m_type in np.unique(target.descri['m_type'])}
        return gap

    def forward(self, input, target, weights_gap, weights_l2):
        """ Computes l2 norm. """
        gap = self.compute_gap(input, target, weights_gap)
        if weights_l2 is None:
            loss = cplx.modulus(gap).pow(2.0).mean()
        else:
            loss = (weights_l2 * cplx.modulus(gap).pow(2.0)).sum()
        return loss


class MSELossCov(nn.Module):
    def __init__(self):
        super(MSELossCov, self).__init__()
        self.max_gap = {}

    def forward(self, input, target):
        gap = torch.zeros_like(target.y)

        # exp
        gap[target.descri.where(q=1), :] = cplx.mul(target.select(q=1), input.select(q=1) - target.select(q=1))

        # cov
        gap[target.descri.where(q=2), :] = input.select(q=2) - target.select(q=2)

        self.max_gap = {m_type: torch.max(cplx.modulus(gap[target.descri.where(m_type=m_type)])).item()
                        for m_type in np.unique(target.descri['m_type'])}

        return cplx.modulus(gap).pow(2.0).mean()
