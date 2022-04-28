import math
from math import factorial as fct
import numpy as np
from kymatio.scattering1d import filter_bank as fb
from numba import jit


def compute_anti_aliasing_filt(N, p):
    freq = np.fft.fftfreq(N)
    freq[N // 2:] += 1
    aaf = np.ones_like(freq)  # initialize Anti Aliasing Filter
    idx = freq >= 0.75
    freq = freq[idx] * 4 - 3.
    freq_pow = np.power(freq, p)
    acc, a = np.zeros_like(freq), 0
    for k in range(p + 1):
        freq_pow *= freq
        bin = fct(p) / (fct(k) * fct(p - k))
        sgn = 1. if k % 2 == 0 else -1.
        acc += bin * sgn * freq_pow / (p + k + 1)
        a += bin * sgn / (p + k + 1)
    dom_coeff = -1 / a
    aaf[idx] = dom_coeff * acc + 1.
    return aaf


def compute_morlet_parameters(J, Q, max_frequency):
    """Compute central frequencies and bandwidth of morlet dictionary."""
    scaling = np.logspace(0, -J, num=J * Q, endpoint=False, base=2)
    mu_freq = max_frequency * scaling
    sigma_freq = np.array([fb.compute_sigma_psi(m, Q) for m in mu_freq])
    return mu_freq, sigma_freq


def compute_morlet_low_pass_parameters(J, Q, max_frequency):
    """
    Compute central frequency and bandwidth of low-pass filter.
    This function uses the oracle sigma_{low} = sigma_J * (2.372 Q + 1.109) where sigma_{low}
    is the bandwidth of the low pass filter, sigma_J is the bandwidth of a morlet wavelet centered in max_frequency
    2^{-J} and :math:Q is the number of wavelet per scales. This function was empiricaly found to minimise the
    dictionary's Littlewood-Paley inequality constant.
    """
    mu_J = max_frequency * 2 ** -J
    sigma_J = fb.compute_sigma_psi(mu_J, Q)
    sigma_low = sigma_J * (2.372 * Q + 1.109)
    return sigma_low


@jit
def compute_battle_lemarie_parameters(N, Q, high_freq=0.5):
    xi_curr = high_freq
    xi, sigma = [], [0.1]
    sigma.pop(0)  # (bad) Trick to enable numba to inference type of sigma
    factor = 1. / math.pow(2., 1. / Q)
    for nq in range(N * Q):
        xi.append(xi_curr)
        xi_curr *= factor

    return xi, sigma


# BL_XI0 = 0.7593990773014584
BL_XI0 = 0.75 * 1.012470304985129


# @jit
def battle_lemarie_psi(N, Q, xi, normalize):
    # if Q != 1:
    #     raise NotImplementedError("Scaling battle-lemarie wavelets to multiple wavelets per octave not implemented yet.")
    xi0 = BL_XI0  # mother wavelet center

    # frequencies for mother wavelet with 1 wavelet per octave
    abs_freqs = np.linspace(0, 1, N + 1)[:-1]
    # frequencies for wavelet centered in xi with 1 wavelet1 per octave
    freqs = abs_freqs * xi0 / xi
    # frequencies for wavelet centered in xi with Q wavelets per octave
    # freqs = xi0 + (xi_freqs - xi0) * Q

    num, den = b_function(freqs)
    num2, den2 = b_function(freqs / 2)
    numpi, denpi = b_function(freqs / 2 + 0.5)

    stable_den = np.empty_like(freqs)
    stable_den[freqs != 0] = np.sqrt(den[freqs != 0]) / (2 * np.pi * freqs[freqs != 0]) ** 4
    # protection in omega = 0
    stable_den[freqs == 0] = 2 ** (-4)

    mask = np.mod(freqs, 2) != 1
    stable_den[mask] *= np.sqrt(den2[mask] / denpi[mask])
    mask = np.mod(freqs, 2) == 1
    # protection in omega = 2pi [4pi]
    stable_den[mask] = np.sqrt(den2[mask]) / (np.pi * freqs[mask]) ** 4

    psi_hat = np.sqrt(numpi / (num * num2)) * stable_den
    psi_hat[freqs < 0] = 0

    # remove small bumps after the main bumps in order to improve the scaling of high frequency wavelets
    idx = 0
    while True:
        idx += 1
        if idx == psi_hat.size - 1 or (psi_hat[idx-1] > psi_hat[idx] and psi_hat[idx] < psi_hat[idx+1]):
            break

    psi_hat[idx+1:] = 0.0

    if normalize == 'l1':
        pass
    if normalize == 'l2':
        psi_hat /= np.sqrt((np.abs(psi_hat) ** 2).sum())

    return psi_hat


#   @jit
def battle_lemarie_phi(N, Q, xi_min):
    xi0 = BL_XI0  # mother wavelet center

    abs_freqs = np.fft.fftfreq(N)
    freqs = abs_freqs * xi0 / xi_min
    # freqs = xi_freqs * Q

    num, den = b_function(freqs)

    stable_den = np.empty_like(freqs)
    stable_den[freqs != 0] = np.sqrt(den[freqs != 0]) / (2 * np.pi * freqs[freqs != 0]) ** 4
    stable_den[freqs == 0] = 2 ** (-4)

    phi_hat = stable_den / np.sqrt(num)
    return phi_hat


@jit
def b_function(freqs, eps=1e-7):
    cos2 = np.cos(freqs * np.pi) ** 2
    sin2 = np.sin(freqs * np.pi) ** 2

    num = 5 + 30 * cos2 + 30 * sin2 * cos2 + 70 * cos2 ** 2 + 2 * sin2 ** 2 * cos2 + 2 * sin2 ** 3 / 3
    num /= 105 * 2 ** 8
    sin8 = sin2 ** 4

    return num, sin8


def compute_bump_steerable_parameters(N, Q, high_freq=0.5):
    return compute_battle_lemarie_parameters(N, Q, high_freq=high_freq)


def low_pass_constants(Q):
    """Function computing the ideal amplitude and variance for the low-pass of a bump
    wavelet dictionary, given the number of wavelets per scale Q.
    The amplitude and variance are computed by minimizing the frame error eta:
        1 - eta <= sum psi_la ** 2 <= 1 + eta
    Simple models are then fitted to compute those values quickly.
    The computation was done using gamma = 1.
    """
    ampl = -0.04809858889110362 + 1.3371665071917382 * np.sqrt(Q)
    xi2sigma = np.exp(-0.35365794431968484 - 0.3808886546835562 / Q)
    return ampl, xi2sigma


@jit
def bump_steerable_psi(N, Q, xi):
    abs_freqs = np.linspace(0, 1, N + 1)[:-1]
    psi = hwin((abs_freqs - xi) / xi, 1.)

    return psi


# @jit
# def bump_steerable_psi(N, Q, xi):
#     sigma = xi * BS_xi2sigma

#     abs_freqs = np.linspace(0, 1, N + 1)[:-1]
#     psi = hwin((abs_freqs - xi) / sigma, 1.)

#     return psi


@jit
def bump_steerable_phi(N, Q, xi_min):
    ampl, xi2sigma = low_pass_constants(Q)
    sigma = xi_min * xi2sigma

    abs_freqs = np.abs(np.fft.fftfreq(N))
    phi = ampl * np.exp(- (abs_freqs / (2 * sigma)) ** 2)

    return phi


# @jit
# def bump_steerable_phi(N, Q, xi_min):
#     sigma = xi_min * BS_xi2sigma

#     abs_freqs = np.abs(np.fft.fftfreq(N))
#     phi = hwin(abs_freqs / sigma, 1.)

#     return phi


@jit
def hwin(freqs, gamma1):
    psi_hat = np.zeros_like(freqs)
    idx = np.abs(freqs) < gamma1

    psi_hat[idx] = np.exp(1. / (freqs[idx] ** 2 - gamma1 ** 2))
    psi_hat *= np.exp(1 / gamma1 ** 2)

    return psi_hat


@jit
def compute_meyer_parameters(N, Q, high_freq):
    return compute_battle_lemarie_parameters(N, Q, high_freq=high_freq)


@jit
def compute_shannon_parameters(J, Q, high_freq):  # bad code
    xi = [high_freq * 2 ** (-j / Q) for j in range(J * Q)]
    sigma = [om * (2 ** (1 / Q) - 1) / (2 ** (1 / Q) + 1) for om in xi]
    return xi, sigma


def shannon_psi(N, Q, xi, sigma):
    freqs = np.linspace(0.0, 1.0, N, endpoint=True)
    psi = 1.0 * ((freqs < xi + sigma) & (xi - sigma <= freqs))
    return psi


def shannon_phi(N, sigma):
    freqs = np.linspace(0.0, 1.0, N, endpoint=True)
    phi = 1.0 * ((freqs <= sigma) | (1.0 - freqs <= sigma))
    return phi


@jit
def meyer_psi(N, Q, xi):
    # if Q != 1:
    #    raise NotImplementedError("Scaling Meyer wavelets to multiple wavelets per octave not implemented yet.")

    # frequencies for mother wavelet with 1 wavelet per octave
    abs_freqs = np.linspace(-0.5, 0.5, N + 1)[:-1]
    psi = meyer_mother_psi(8 / 3 * np.pi * (abs_freqs) / xi)
    return np.fft.fftshift(psi)


@jit
def meyer_phi(N, Q, xi):
    abs_freqs = np.linspace(-0.5, 0.5, N + 1)[:-1]
    phi = meyer_mother_phi(8 / 3 * np.pi * (abs_freqs) / xi)
    return np.fft.fftshift(phi)


def nu(x):
    out = np.zeros(x.shape)
    idx = np.logical_and(0 < x, x < 1)
    out[idx] = x[idx] ** 4 * (35 - 84 * x[idx] + 70 * x[idx] ** 2 - 20 * x[idx] ** 3)
    return out


def meyer_mother_psi(w):
    psi = np.zeros(w.shape) + 1j * np.zeros(w.shape)
    idx = np.logical_and(2 * np.pi / 3 < w, w < 4 * np.pi / 3)
    psi[idx] = np.sin(np.pi / 2 * nu(3 * np.abs(w[idx]) / 2 / np.pi - 1)) / np.sqrt(2 * np.pi)  # * np.exp(1j*w[idx]/2)

    idx = np.logical_and(4 * np.pi / 3 < w, w < 8 * np.pi / 3)
    psi[idx] = np.cos(np.pi / 2 * nu(3 * np.abs(w[idx]) / 4 / np.pi - 1)) / np.sqrt(2 * np.pi)  # * np.exp(1j*w[idx]/2)

    return 2 * psi


def meyer_mother_phi(w):
    phi = np.zeros(w.shape) + 1j * np.zeros(w.shape)
    idx = np.abs(w) < 2 * np.pi / 3
    phi[idx] = 1 / np.sqrt(2 * np.pi)
    idx = np.logical_and(2 * np.pi / 3 < np.abs(w), np.abs(w) < 4 * np.pi / 3)
    phi[idx] = np.cos(np.pi / 2 * nu(3 * np.abs(w[idx]) / 2 / np.pi - 1)) / np.sqrt(2 * np.pi)
    return phi * 2
