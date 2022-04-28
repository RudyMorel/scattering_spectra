import warnings
import numpy as np
from numpy.random import normal as nd


def gaussian_cme(cov, R, T):
    """
    Create S synthesis of a gaussian process of length T with the specified
    autocovariance through circulant matrix embedding (see C. R. Dietrich AND G. N. Newsam).

    :param cov: T array, function r such that Cov[Y(x)Y(y)] = r(|x-y|)
    :param: S: int, number of synthesis
    :param T: int, number of samples
    """

    # Circulant matrix embedding: fft of periodized autocovariance:
    cov = np.concatenate((cov, np.flip(cov[1:-1])), axis=0)
    L = np.fft.fft(cov)[None, :]
    if np.any(L.real < 0):
        warnings.warn('Found FFT of covariance < 0. Embedding matrix is not non-negative definite.')

    # Random noise in Fourier domain
    z = np.random.randn(R, 2 * T - 2) + 1j * np.random.randn(R, 2 * T - 2)

    # Impose covariance and invert
    # Use fft to ignore normalization, because only real part is needed.
    x = np.fft.fft(z * np.sqrt(L / (2 * T - 2)), axis=-1).real

    # First N samples have autocovariance cov:
    x = x[:, :T]

    return x


def powerSpectrumNormalization(x, K):
    """
    A function that enables the TS x to match the power spectrum K by convolution

    :param x: The vector to normalize
    :param K: The targeted power spectrum
    :return: The vector x\circ kappa with same spectrum as K
    """
    Kfft = np.fft.fft(K)
    xfft = np.fft.fft(x)

    discrepancy = np.divide(np.abs(Kfft), np.abs(xfft))
    x_normalized_fft = np.multiply(xfft, discrepancy)

    return np.fft.ifft(x_normalized_fft).real


def fbm(R, T, H, sigma=1, dt=None):
    """
    Create a realization of fractional Brownian motion using circulant
    matrix embedding.

    Inputs:
      - shape: if scalar, it is the  number of samples. If tuple it is (N, R), the
               number of samples and realizations, respectively.
      - H (scalar): Hurst exponent.
      - sigma (scalar): variance of processr

    Outputs:
      - fbm: synthesized fbm realizations. If 'shape' is scalar, fbm is of shape (N,).
             Otherwise, it is of shape (N, R).
    """
    if not 0 <= H <= 1:
        raise ValueError('H must satisfy 0 <= H <=1')

    if not dt:
        dt = 1 / T

    # Create covariance of fGn
    n = np.arange(T)
    r = dt ** (2 * H) * sigma ** 2 / 2 * (
                np.abs(n + 1) ** (2 * H) + np.abs(n - 1) ** (2 * H) - 2 * np.abs(n) ** (2 * H))

    fbm = np.cumsum(gaussian_cme(r, R, T), axis=1)

    return fbm


def geom_brownian(R: int, T: float, nb_sample: int, S0: float, mu: float, sigma: float):
    """Simulate a geometric brownian also called Black-Scholes trajectory of adimensional trend mu and vol sigma."""
    B = np.cumsum(np.random.randn(R, nb_sample), -1) / np.sqrt(nb_sample)
    X = S0 * np.exp((mu - 0.5 * sigma ** 2) * np.arange(T) + sigma * np.sqrt(T) * (B - B[:, 0:1]))

    return X
