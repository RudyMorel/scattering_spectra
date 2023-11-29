""" Poisson process. """
import numpy as np
from scipy.stats import poisson


def poisson_t(B: int, T: int, N: int, signed: bool) -> np.ndarray:
    """ Poisson process on [0,T] with n jumps.

    :param B: batch size, number of generrated time-series
    :param T: time-series number of steps
    :param N: number of jumps
    :param signed: if True, jumps are signed (+1/-1)
    """
    x = np.zeros((B, T))
    positions = np.random.rand(B,T).argsort(axis=-1)[:, :N]
    sign = 2 * (np.random.rand(N) > (0.5 if signed else 0.0)) - 1
    x[np.arange(B)[:,None], positions] = sign
    return x.cumsum(-1)


def poisson_mu(B: int, T: int, mu: float, signed: bool):
    """ Poisson process on [0,T] with intensity mu.

    :param B: batch size, number of generrated time-series
    :param T: time-series number of steps
    :param mu: jump process intensity
    :param signed: if True, jumps are signed (+1/-1)
    """
    N = poisson.rvs(T * mu)
    x = poisson_t(B, T, min(N, T), signed)
    return x
