""" General usefull nn Modules"""
import numpy as np
import torch
import torch.nn as nn


class NormalizationLayer(nn.Module):
    """ Divide certain dimension by specified values. """
    def __init__(
        self,
        dim: int,
        sigma: torch.Tensor | None,
        on_the_fly: bool
    ) -> None:
        super(NormalizationLayer, self).__init__()
        self.dim = dim
        if sigma is not None and ((sigma <= 0).any() or torch.isnan(sigma).any()):
            raise ValueError("Sigma must contain only positive values.")
        self.register_buffer("sigma", sigma)
        self.on_the_fly = on_the_fly

    def forward(
        self, 
        x: torch.Tensor, 
        bs: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.on_the_fly:  # normalize on the fly
            sigma = torch.abs(x).pow(2.0).mean(-1,keepdim=True).pow(0.5)
            return x / sigma
        if self.sigma is None:
            return x
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
        # register as a buffer (not a plain attribute) so that .to(device),
        # .cuda(), .float(), .double() etc. correctly move/cast it
        self.register_buffer(
            "phases", torch.cos(phases) + 1j * torch.sin(phases)
        )

    def _apply(self, fn, *args, **kwargs):
        # The phases buffer is complex128, which the MPS backend cannot hold
        # (no float64/complex128 support). When this module is moved to such a
        # device, downcast the buffer to complex64 first so the move succeeds.
        # cpu/cuda are untouched, so their precision is unchanged.
        probe = fn(torch.zeros(1, dtype=torch.float32))
        if probe.device.type == 'mps' and self.phases.dtype == torch.complex128:
            self.phases = self.phases.to(torch.complex64)
        return super()._apply(fn, *args, **kwargs)

    def forward(self, x):
        """ Computes Re(e^{i alpha} x) for alpha in self.phases. """
        return (self.phases[..., :, None] * x).real


class LinearLayer(nn.Module):
    
    def __init__(self, L: torch.Tensor) -> None:
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