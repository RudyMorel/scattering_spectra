""" General usefull nn Modules"""
import numpy as np
import torch
import torch.nn as nn


class NormalizationLayer(nn.Module):
    def __init__(self, dim: int, sigma2: torch.tensor, on_the_fly: bool):
        super(NormalizationLayer, self).__init__()
        self.dim = dim
        self.sigma2 = sigma2
        self.on_the_fly = on_the_fly

    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.on_the_fly:  # normalize on the fly
            sigma2 = torch.abs(x).pow(2.0).mean(-1, keepdim=True)
            return x / sigma2.pow(0.5)
        return x / self.sigma2[(..., *(None,) * (x.ndim - 1 - self.dim))]


class SkipConnection(nn.Module):
    """ Skip connection. """
    def __init__(self, module: nn.Module, dim: int):
        super(SkipConnection, self).__init__()
        self.module = module
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.module(x)
        return torch.cat([x, y], dim=self.dim)


class Modulus(nn.Module):
    """ Modulus. """
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class ChunkedModule(nn.Module):
    """ Modulus. """
    def __init__(self, nchunks: int, module: nn.Module):
        super(ChunkedModule, self).__init__()
        self.nchunks = nchunks
        self.module = module

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Chunked forward on the batch dimension.

        :param x: B x ... tensor
        :return:
        """
        batch_split = np.array_split(np.arange(x.shape[0]), self.nchunks)
        output_l = []
        for bs in batch_split:
            output_l.append(self.module(x[bs, ...]))
        return torch.cat(output_l)
