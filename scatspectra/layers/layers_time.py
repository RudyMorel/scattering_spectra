""" Implements layers that operate on time. """
from packaging import version
import numpy as np
import torch
import torch.nn as nn

from scatspectra.layers import init_band_pass, init_low_pass, ScaleIndexer


# Fourier and torch compatibility
if version.parse(torch.__version__) >= version.parse('1.8'):
    fft = torch.fft.fft
    ifft = torch.fft.ifft
else:
    def fft(x):
        if torch.is_floating_point(x):
            x = torch.complex(x, torch.zeros_like(x))
        return torch.view_as_complex(torch.fft(torch.view_as_real(x), 1, normalized=False))

    def ifft(x):
        if torch.is_floating_point(x):
            x = torch.complex(x, torch.zeros_like(x))
        return torch.view_as_complex(torch.ifft(torch.view_as_real(x), 1, normalized=False))


class Pad1d(nn.Module):
    """ Padding base class. """
    
    def __init__(self, T: int) -> None:
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

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, torch.flip(x, dims=(-1,))], dim=-1)

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :self.T]


class Wavelet(nn.Module):
    """ Wavelet convolutional operator. """

    def __init__(self, T: int, J: int, Q: int,
                 wav_type: str, wav_norm: str, high_freq: float, rpad: bool,
                 layer_r: int,
                 sc_idxer: ScaleIndexer):
        super(Wavelet, self).__init__()
        self.T, self.J, self.Q, self.layer_r = T, J, Q, layer_r
        if rpad:
            self.T *= 2
        self.wav_type, self.high_freq, self.wav_norm = wav_type, high_freq, wav_norm
        self.sc_idxer = sc_idxer

        psi_hat = init_band_pass(wav_type, self.T, J, Q, high_freq, wav_norm)
        phi_hat = init_low_pass(wav_type, self.T, J, Q, high_freq)
        filt_hat = np.concatenate([psi_hat, phi_hat[None, :]])
        self.register_buffer('filt_hat', torch.tensor(filt_hat))

        self.Pad = ReflectionPad(T) if rpad else Pad1d(T)

        self.pairing = self.get_pairing()

    def get_pairing(self):
        """ Initialize pairing to avoid computing negligable convolutions. """

        if self.layer_r == 1:
            return np.array([(0, j) for j in range(self.sc_idxer.JQ(r=1) + 1)])

        pairs = []
        for path in self.sc_idxer.sc_paths[self.layer_r - 1]:
            parent_path = path[:-1]
            idx = [i for i, p in enumerate(self.sc_idxer.sc_paths[self.layer_r - 2]) if (p == parent_path).all()][0]
            pairs.append((idx, path[-1]))

        pairing = np.array(pairs)

        return pairing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Performs in Fourier the convolution (x * psi_lam, x * phi_J).

        :param x: (C) x Jr x A x T tensor
        :return: (C) x J{r+1} x A x T tensor
        """

        # since idx[:,0] is always lower than x_pad.shape[2], doing fft in second is always optimal
        x_pad = self.Pad.pad(x)
        x_hat = fft(x_pad)
        x_filt_hat = x_hat[..., self.pairing[:, 0], :, :] * self.filt_hat[self.pairing[:, 1], :].unsqueeze(-2)
        x_filt = self.Pad.unpad(ifft(x_filt_hat))

        return x_filt
