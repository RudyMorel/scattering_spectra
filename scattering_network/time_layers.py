import numpy as np
import torch
import torch.nn as nn

import utils.complex_utils as cplx
import scattering_network.filter_bank as lfb


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


fft = FFT(lambda x: torch.view_as_real(torch.fft.fft(torch.view_as_complex(x))),
          lambda x: torch.view_as_real(torch.fft.ifft(torch.view_as_complex(x))),
          lambda x: torch.fft.ifft(torch.view_as_complex(x)).real, type_checks)


def fft1d_c2c(x):
    return fft(x, direction='C2C', inverse=False)


def ifft1d_c2c_normed(x):
    return fft(x, direction='C2C', inverse=True)


class Pad1d(nn.Module):
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
    """
    Reflection pad.
    """
    def __init__(self, T):
        super(ReflectionPad, self).__init__(T)
        unpad_idx = np.arange(T)
        self.register_buffer('unpad_idx', torch.tensor(unpad_idx, dtype=torch.int64))

    def pad(self, x):
        return torch.cat([x, torch.flip(x, dims=(-2,))], dim=-2)
        # return torch.cat([x, torch.flip(x[..., 1:-1, :], dims=(-2,))], dim=-2)

    def unpad(self, x):
        # return torch.split(x, self.T, dim=-2)[0]
        return x[..., :self.T, :]


class TimeConvolutionPadding(nn.Module):
    def __init__(self, psi_hat, phi_hat, Pad):
        super(TimeConvolutionPadding, self).__init__()
        self.register_buffer('filt_hat', torch.cat([psi_hat, phi_hat, phi_hat], dim=0))
        self.Pad = Pad

    def forward(self, x, idx):
        """
        Performs in Fourier the convolution x * psi_lam.

        :param x: (C) x Jr x A x T x 2 tensor
        :param idx: CJ x 2 array
        :return: J{r+1} x T x 2 tensor
        """
        # since idx[:,0] is always lower than x_pad.shape[2], doing fft in second is always optimal
        x_pad = self.Pad.pad(x)
        x_hat = fft1d_c2c(x_pad)
        x_filt_hat = cplx.mul(x_hat[..., idx[:, 0], :, :, :], self.filt_hat[idx[:, 1], :, :].unsqueeze(-3))
        x_filt = self.Pad.unpad(ifft1d_c2c_normed(x_filt_hat))
        return x_filt


class TimeConvolutionReflectionPad(TimeConvolutionPadding):
    def __init__(self, T, psi_hat, phi_hat):
        super(TimeConvolutionReflectionPad, self).__init__(psi_hat, phi_hat, ReflectionPad(T))


class Wavelet(TimeConvolutionReflectionPad):
    """
    Wavelet operator.
    """
    def __init__(self, T, J, Q, wav_type, high_freq, wav_norm):

        self.T, self.J, self.Q = T + T, J, Q
        self.wav_type, self.high_freq, self.wav_norm = wav_type, high_freq, wav_norm

        self.xi, self.sigma = None, None
        psi_hat, phi_hat = self.init_band_pass(), self.init_low_pass()

        super(Wavelet, self).__init__(T, psi_hat, phi_hat)

    def init_band_pass(self):
        Q = self.Q
        high_freq = self.high_freq

        # initialize wavelet parameters
        if self.wav_type == 'morlet':
            xi, sigma = lfb.compute_morlet_parameters(self.J, self.Q, high_freq)
        elif self.wav_type == 'battle_lemarie':
            # if Q != 1:
            #     print("\nWarning: width of Battle-Lemarie wavelets not adaptative with Q.\n")
            xi, sigma = lfb.compute_battle_lemarie_parameters(self.J, self.Q, high_freq=high_freq)
        elif self.wav_type == 'bump_steerable':
            if Q != 1:
                print("\nWarning: width of Bump-Steerable wavelets not adaptative with Q.\n")
            xi, sigma = lfb.compute_bump_steerable_parameters(self.J, self.Q, high_freq=high_freq)
        elif self.wav_type == 'meyer':
            # if Q != 1:
            #    print("\nWarning: width of Meyer wavelets not adaptative with Q in the current implementation.\n")
            xi, sigma = lfb.compute_meyer_parameters(self.J, self.Q, high_freq=high_freq)
        elif self.wav_type == 'shannon':
            xi, sigma = lfb.compute_shannon_parameters(self.J, self.Q, high_freq=high_freq)
        else:
            raise ValueError("Unkown wavelet type: {}".format(self.wav_type))
        self.xi = np.array(xi)
        self.sigma = np.array(sigma)

        # initialize wavelets
        if self.wav_type == "morlet":
            psi_hat = [lfb.morlet_1d(self.T, xi, sigma, self.wav_norm) for xi, sigma in zip(self.xi, self.sigma)]
        elif self.wav_type == "battle_lemarie":
            psi_hat = [lfb.battle_lemarie_psi(self.T, self.Q, xi, self.wav_norm) for xi in self.xi]
            # psi_hat[]
        elif self.wav_type == "bump_steerable":
            psi_hat = [lfb.bump_steerable_psi(self.T, xi) / np.sqrt(self.Q) for xi in self.xi]
        elif self.wav_type == 'meyer':
            psi_hat = [lfb.meyer_psi(self.T, 1, xi) for xi in self.xi]
        elif self.wav_type == 'shannon':
            psi_hat = [lfb.shannon_psi(self.T, xi, sigma) for xi, sigma in zip(self.xi, self.sigma)]
        else:
            raise ValueError("Unrecognized wavelet type.")
        psi_hat = np.stack(psi_hat, axis=0)

        # some high frequency wavelets have strange behavior at negative low frequencies
        psi_hat[:, -self.T // 8:] = 0.0

        # pytorch parameter
        psi_hat = nn.Parameter(cplx.from_np(psi_hat), requires_grad=False)
        return psi_hat

    def init_low_pass(self):
        """Compute the low-pass Fourier transforms assuming it has the same variance
        as the lowest-frequency wavelet.
        """
        self.xi = np.append(self.xi, 0.0)
        if self.wav_type == "morlet":
            sigma_low = self.sigma[-1]
            np.append(self.sigma, lfb.compute_morlet_low_pass_parameters(self.J, self.Q, self.high_freq))
            phi_hat = lfb.gauss_1d(self.T, sigma_low)
        elif self.wav_type == "battle_lemarie":
            xi_low = self.xi[-2]  # Because 0 was appended for Morlet
            # phi_hat = lfb.battle_lemarie_phi(2*self.T, self.Q, xi_low)
            phi_hat = lfb.battle_lemarie_phi(self.T, 1, xi_low)
        elif self.wav_type == "bump_steerable":
            xi_low = self.xi[-2]  # Because 0 was appended for Morlet
            phi_hat = lfb.bump_steerable_phi(self.T, 1, xi_low)
        elif self.wav_type == 'meyer':
            xi_low = self.xi[-2]
            phi_hat = lfb.meyer_phi(self.T, xi_low)
        elif self.wav_type == 'shannon':
            sigma_low = self.xi[-2] - self.sigma[-1]
            phi_hat = lfb.shannon_phi(self.T, sigma_low)
        else:
            raise ValueError("Unrecognized wavelet type.")

        # pytorch parameter
        phi_hat = cplx.from_np(phi_hat).unsqueeze(0)
        return phi_hat
