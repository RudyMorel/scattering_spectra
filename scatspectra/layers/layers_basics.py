""" General usefull nn Modules"""
import numpy as np
import torch
import torch.nn as nn


class NormalizationLayer(nn.Module):
    """ Divide certain dimension by specified values. """
    def __init__(self,
                 dim: int,
                 sigma: torch.Tensor | None,
                 on_the_fly: bool) -> None:
        super(NormalizationLayer, self).__init__()
        self.dim = dim
        self.sigma = sigma
        self.on_the_fly = on_the_fly

    def forward(self, x: torch.Tensor, 
                bs: torch.Tensor | None = None) -> torch.Tensor:
        if self.sigma is None:
            return x
        if self.on_the_fly:  # normalize on the fly
            sigma = torch.abs(x).pow(2.0).mean(-1,keepdim=True).pow(0.5)
            return x / sigma
        sigma = self.sigma[(..., *(None,) * (x.ndim - 1 - self.dim))]
        if bs is not None and self.sigma.shape[0] > 1:
            sigma = sigma[bs,...]
        return x / sigma


class Modulus(nn.Module):
    """ Modulus. """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class PhaseOperator(nn.Module):
    """ Sample complex phases and creates complex phase channels. """

    def __init__(self, A: int):
        super(PhaseOperator, self).__init__()
        phases = torch.tensor(np.linspace(0, np.pi, A, endpoint=False))
        self.phases = torch.cos(phases) + 1j * torch.sin(phases)

    def cpu(self):
        self.phases = self.phases.cpu()
        return self

    def cuda(self):
        self.phases = self.phases.cuda()
        return self

    def forward(self, x):
        """ Computes Re(e^{i alpha} x) for alpha in self.phases. """
        return (self.phases[..., :, None] * x).real


class LinearLayer(nn.Module):
    def __init__(self,
                 L: torch.Tensor) -> None:
        super(LinearLayer, self).__init__()
        self.register_buffer("L", L)

    def forward(self, x: torch.Tensor, c1=-4, c2=-1):
        """
        Perform Lx a linear transform on x along certain dimension.

        :param x: tensor (B) x T
        :param c1: the index of the 1st dimension to take product on
        :param c2: the index of the 2nd dimension to take product on
        :return:
        """
        L = self.L
        if x.is_complex():
            L = torch.complex(self.L, torch.zeros_like(self.L))
        c1p = c2 if c1 % x.shape[0] == -1 else c1
        x_temp = x.transpose(c2, -1).transpose(c1p, -2)
        return (L @ x_temp).transpose(c1p, -2).transpose(c2, -1)