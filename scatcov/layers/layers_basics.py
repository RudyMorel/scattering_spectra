""" General usefull nn Modules"""
import numpy as np
import torch
import torch.nn as nn

from scatcov.layers.described_tensor import DescribedTensor


class NormalizationLayer(nn.Module):
    """ Divide certain dimension by specified values. """
    def __init__(self,
                 dim: int,
                 sigma: torch.tensor,
                 on_the_fly: bool) -> None:
        super(NormalizationLayer, self).__init__()
        self.dim = dim
        self.sigma = sigma
        self.on_the_fly = on_the_fly

    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.on_the_fly:  # normalize on the fly
            sigma = torch.abs(x).pow(2.0).mean(-1, keepdim=True).pow(0.5)
            return x / sigma
        return x / self.sigma[(..., *(None,) * (x.ndim - 1 - self.dim))]


class ChunkedModule(nn.Module):
    """ Modulus. """
    def __init__(self,
                 module: nn.Module,
                 nchunks: int) -> None:
        super(ChunkedModule, self).__init__()
        self.nchunks = nchunks
        self.module = module

    def forward(self, x: torch.tensor) -> DescribedTensor:
        """
        Chunked forward on the batch dimension.

        :param x: B x ... tensor
        :return:
        """
        batch_split = np.array_split(np.arange(x.shape[0]), self.nchunks)
        Rxs = []
        for bs in batch_split:
            Rx_chunked = self.module(x[bs, ...])
            Rxs.append(Rx_chunked)
        return DescribedTensor(x=None, y=torch.cat([Rx.y for Rx in Rxs]), descri=Rxs[-1].descri)
