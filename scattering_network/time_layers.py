""" Implements layers that operate on time. """
from typing import *
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
from packaging import version

import utils.complex_utils as cplx
from utils import multid_where_np
from scattering_network.filter_bank import init_band_pass, init_low_pass
from scattering_network.module_chunk import SubModuleChunk
from scattering_network.described_tensor import Description, DescribedTensor


class FFT:
    def __init__(self, fft, ifft, irfft, type_checks):
        self.fft = fft
        self.ifft = ifft
        self.irfft = irfft
        self.sanity_checks = type_checks

    def fft_forward(self, x, direction='C2C', inverse=False):
        """Interface with FFT routines for any dimensional signals and any backend signals.

            Example (for Torch)
            -------
            x = torch.randn(128, 32, 32, 2)
            x_fft = fft(x)
            x_ifft = fft(x, inverse=True)

            Parameters
            ----------
            x : input
                Complex input for the FFT.
            direction : string
                'C2R' for complex to real, 'C2C' for complex to complex.
            inverse : bool
                True for computing the inverse FFT.
                NB : If direction is equal to 'C2R', then an error is raised.

            Raises
            ------
            RuntimeError
                In the event that we are going from complex to real and not doing
                the inverse FFT or in the event x is not contiguous.


            Returns
            -------
            output :
                Result of FFT or IFFT.
        """
        if direction == 'C2R':
            if not inverse:
                raise RuntimeError('C2R mode can only be done with an inverse FFT.')

        self.sanity_checks(x)

        if direction == 'C2R':
            output = self.irfft(x)
        elif direction == 'C2C':
            if inverse:
                output = self.ifft(x)
            else:
                output = self.fft(x)
        else:
            raise ValueError("Unrecognized direction in fft.")

        return output

    def __call__(self, x, direction='C2C', inverse=False):
        return self.fft_forward(x, direction=direction, inverse=inverse)


def _is_complex(x):
    return x.shape[-1] == 2


def type_checks(x):
    if not _is_complex(x):
        raise TypeError('The input should be complex (i.e. last dimension is 2).')

    if not x.is_contiguous():
        raise RuntimeError('Tensors must be contiguous.')


if version.parse(torch.__version__) >= version.parse('1.8'):
    fft = FFT(lambda x: torch.view_as_real(torch.fft.fft(torch.view_as_complex(x))),
              lambda x: torch.view_as_real(torch.fft.ifft(torch.view_as_complex(x))),
              lambda x: torch.fft.ifft(torch.view_as_complex(x)).real, type_checks)
else:
    fft = FFT(lambda x: torch.fft(x, 1, normalized=False),
              lambda x: torch.ifft(x, 1, normalized=False),
              lambda x: torch.irfft(x, 1, normalized=False, onesided=False),
              type_checks)


def fft1d_c2c(x):
    """ The Fourier transform on complex tensors. """
    return fft(x, direction='C2C', inverse=False)


def ifft1d_c2c_normed(x):
    """ The inverse Fourier transform on complex tensors. """
    return fft(x, direction='C2C', inverse=True)


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
    def __init__(self, T):
        super(ReflectionPad, self).__init__(T)
        unpad_idx = np.arange(T)
        self.register_buffer('unpad_idx', torch.tensor(unpad_idx, dtype=torch.int64))

    def pad(self, x):
        return torch.cat([x, torch.flip(x, dims=(-2,))], dim=-2)
        # return torch.cat([x, torch.flip(x[..., 1:-1, :], dims=(-2,))], dim=-2)

    def unpad(self, x):
        return x[..., :self.T, :]


class Wavelet(SubModuleChunk):
    """ Wavelet convolutional operator. """
    def __init__(self, T, J, Q, wav_type, high_freq, wav_norm, layer_r, sc_idxer):
        super(Wavelet, self).__init__()
        self.T, self.J, self.Q, self.layer_r = 2 * T, J, Q, layer_r
        self.wav_type, self.high_freq, self.wav_norm = wav_type, high_freq, wav_norm
        self.sc_idxer = sc_idxer

        psi_hat = init_band_pass(wav_type, self.T, J, Q, high_freq, wav_norm)
        phi_hat = init_low_pass(wav_type, self.T, J, Q, high_freq)
        self.filt_hat = nn.Parameter(cplx.from_np(np.concatenate([psi_hat, phi_hat[None, :]])), requires_grad=False)
        self.Pad = ReflectionPad(T)

        # params
        self.masks = []
        self.sc_pairing = []

    def external_surjection_aux(self, input_descri):
        """ Return description that can be computed on input_descri. """
        n, r, sc, *js, a, low = input_descri
        descri_l = []
        out_columns = ['n1', 'r', 'sc'] + [f'j{r}' for r in range(1, self.sc_idxer.r_max + 1)] + ['a', 'low']

        # from path j1, ..., j{r-1} associate all paths j1, ..., j{r-1}, jr
        for jrp1 in range(0 if r == 0 else js[r-1] + 1, self.sc_idxer.JQ() + 1):
            row = [-1] * (5 + self.sc_idxer.r_max)

            js_here = self.sc_idxer.idx_to_path(sc)
            js_here += (jrp1, )

            # n, r, sc
            row[:3] = n, self.layer_r, self.sc_idxer.path_to_idx(js_here)

            # js
            row[3: 3 + self.layer_r] = js_here

            # a, low
            row[-2:] = a, jrp1 == self.sc_idxer.JQ()

            descri_l.append(namedtuple('Descri', out_columns)(*row))

        return descri_l

    def internal_surjection(self, output_descri_row: NamedTuple) -> List[NamedTuple]:
        """ Return rows that can be computed on output_descri_row. """
        return []

    def init_one_chunk(self, input: DescribedTensor, output_descri: Description, i_chunk: int) -> None:
        """ Init the parameters of the model required to compute output_descri from input. """
        columns_previous_layer = ['n1'] + [f'j{r}' for r in range(1, self.layer_r)]
        n_j1 = output_descri.to_array(columns_previous_layer)
        sc_pairing_left = multid_where_np(n_j1, input.descri.to_array(columns_previous_layer))
        sc_pairing_right = output_descri.to_array([f'j{self.layer_r}'])[:, 0]
        sc_pairing = np.stack([sc_pairing_left, sc_pairing_right], axis=1)

        self.sc_pairing.append(sc_pairing)

    def clear_params(self) -> None:
        self.sc_pairing = []

    def forward_chunk(self, x: torch.tensor, i_chunk: int):
        """ Performs in Fourier the convolution x * psi_lam.

        :param x: (C) x Jr x A x T x 2 tensor
        :return: J{r+1} x T x 2 tensor
        """
        descri = self.descri[i_chunk]
        pairing = self.sc_pairing[i_chunk]

        if self.layer_r > 1:
            x = cplx.from_real(cplx.modulus(x))

        # since idx[:,0] is always lower than x_pad.shape[2], doing fft in second is always optimal
        x_pad = self.Pad.pad(x)
        x_hat = fft1d_c2c(x_pad)
        x_filt_hat = cplx.mul(x_hat[..., pairing[:, 0], :, :], self.filt_hat[pairing[:, 1], :, :])
        x_filt = self.Pad.unpad(ifft1d_c2c_normed(x_filt_hat))

        return DescribedTensor(x=None, descri=descri, y=x_filt.reshape((x_filt.shape[0], -1) + x_filt.shape[-2:]))


class SpectrumNormalization(SubModuleChunk):
    """ After a wavelet layer, divide each band-pass channel by its energy: X*psi_j ->  X*psi_j / (E|X*psi_j|^2)^0.5 """
    def __init__(self, on_the_fly: Optional[bool] = False, sigma: Optional[DescribedTensor] = None):
        super(SpectrumNormalization, self).__init__()
        if not on_the_fly and sigma is None:
            raise ValueError("SpectrumNormalization requires a power-spectrum to use as normalization.")
        self.on_the_fly = on_the_fly
        self.sigma = sigma

        # params
        self.masks = []

    def external_surjection_aux(self, input_descri: NamedTuple) -> List[NamedTuple]:
        """ Return description that can be computed on input_descri. """
        return [input_descri]

    def internal_surjection(self, output_descri_row: NamedTuple) -> List[NamedTuple]:
        """ Return rows that can be computed on output_descri_row. """
        return []

    def set_sigma(self, sigma: torch.tensor, i_chunk: int) -> None:
        self.register_buffer(f'sigma_{i_chunk}', sigma)

    def clear_params(self) -> None:
        self.masks = []

    def forward_chunk(self, x: torch.tensor, i_chunk: int):
        """ Performs normalization x * psi_lam -> x * psi_lam / sigma_j. """
        descri = self.descri[i_chunk]
        # mask = self.masks[i_chunk]

        if self.on_the_fly:
            sigma = cplx.modulus(x).pow(2.0).mean(-1).pow(0.5).unsqueeze(-1).unsqueeze(-1)
        else:
            if f'sigma_{i_chunk}' not in self.state_dict().keys():
                sigma = cplx.modulus(x).pow(2.0).mean(-1).pow(0.5).unsqueeze(-1).unsqueeze(-1)
                self.set_sigma(sigma, i_chunk)
            sigma = self.state_dict()[f'sigma_{i_chunk}']

        y = x / sigma

        return DescribedTensor(x=None, descri=descri, y=y)
