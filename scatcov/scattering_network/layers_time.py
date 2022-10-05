""" Implements layers that operate on time. """
import numpy as np
import torch
import torch.nn as nn

from scatcov.scattering_network.filter_bank import init_band_pass, init_low_pass
from scatcov.scattering_network.scale_indexer import ScaleIndexer


class Pad1d(nn.Module):
    """ Padding base class. """
    def __init__(self, T):
        super(Pad1d, self).__init__()
        self.T = T

    def pad(self, x):
        return x

    def unpad(self, x):
        return x

    def output_size(self):
        return self.T


class ReflectionPad(Pad1d):
    """ Reflection pad. """
    def pad(self, x: torch.tensor) -> torch.tensor:
        return torch.cat([x, torch.flip(x, dims=(-1,))], dim=-1)

    def unpad(self, x: torch.tensor) -> torch.tensor:
        return x[..., :self.T]


class Wavelet(nn.Module):
    """ Wavelet convolutional operator. """
    def __init__(self, T: int, J: int, Q: int,
                 wav_type: str, high_freq: float, wav_norm: str,
                 layer_r: int, sc_idxer: ScaleIndexer):
        super(Wavelet, self).__init__()
        self.T, self.J, self.Q, self.layer_r = 2 * T, J, Q, layer_r
        self.wav_type, self.high_freq, self.wav_norm = wav_type, high_freq, wav_norm
        self.sc_idxer = sc_idxer

        psi_hat = init_band_pass(wav_type, self.T, J, Q, high_freq, wav_norm)
        phi_hat = init_low_pass(wav_type, self.T, J, Q, high_freq)
        filt_hat = torch.tensor(np.concatenate([psi_hat, phi_hat[None, :]]))
        self.filt_hat = nn.Parameter(filt_hat, requires_grad=False)

        self.Pad = ReflectionPad(T)

        self.pairing = self.get_pairing()

    def get_pairing(self):
        """ Initialize pairing to avoid computing negligable convolutions. """
        # r = 1
        pairing_1 = np.array([(0, j) for j in self.sc_idxer.p_idx[0][:, 0]])

        # r = 2
        pairing_2 = self.sc_idxer.p_idx[1]

        return [pairing_1, pairing_2]

    def forward(self, x: torch.tensor) -> torch.tensor:
        """ Performs in Fourier the convolution (x * psi_lam, x * phi_J).

        :param x: (C) x Jr x A x T tensor
        :return: (C) x J{r+1} x A x T tensor
        """

        pairing = self.pairing[self.layer_r-1]

        # since idx[:,0] is always lower than x_pad.shape[2], doing fft in second is always optimal
        x_pad = self.Pad.pad(x)
        x_hat = torch.fft.fft(x_pad)
        x_filt_hat = x_hat[..., pairing[:, 0], :, :] * self.filt_hat[pairing[:, 1], :].unsqueeze(-2)
        x_filt = self.Pad.unpad(torch.fft.ifft(x_filt_hat))

        return x_filt
