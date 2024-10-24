from typing import Generator
import contextlib
import numpy as np
import torch
from scipy.signal import convolve as sci_convolve


def cumsum_zero(dx: np.ndarray) -> np.ndarray:
    """Cumsum of a vector preserving dimension through zero-pading."""
    res = np.cumsum(dx, axis=-1)
    res = np.concatenate([np.zeros_like(res[..., 0:1]), res], axis=-1)
    return res


def windows(
    x: np.ndarray, 
    w: int, 
    s: int, 
    offset: int = 0, 
    cover_end: bool = False
) -> np.ndarray:
    """ Separate x into windows on last axis, discard any residual. """
    if offset > 0 and cover_end:
        raise ValueError("No offset should be provided if cover_end is True.")
    if offset > 0:
        return windows(x[...,offset:], w, s, 0, cover_end)
    if cover_end:
        offset = x.shape[-1] % w
        return windows(x, w, s, offset, cover_end=False)
    nrows = 1 + (x.shape[-1] - w) // s
    n = x.strides[-1]
    return np.lib.stride_tricks.as_strided(x, shape=x.shape[:-1]+(nrows,w), strides=x.strides[:-1]+(s*n,n))


def shifted_product_aux(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes < x(t-tau) y(t) >_t for values of tau between -T+1 and T-1

    :param x: (B) x T array
    :param y: (B) x T array
    :return: tuple of two arrays
        - first array: tau = 0 to tau = T-1 (x in advance on y)
        - second array: tau = 0 to tau = -T+1 (y in advance on x)
    """
    assert x.shape == y.shape

    # perform convolution
    y = np.flip(y, axis=-1)
    shape = x.shape[:-1]
    x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    conv = np.stack([sci_convolve(x1d, y1d, method='fft') for (x1d, y1d) in zip(x, y)], axis=0)  # batched

    # normalization factors to get averages
    T = x.shape[-1]
    norms = np.array([T - np.abs(tau) for tau in range(-T + 1, T)])

    # the result for every tau
    sft_prod = conv / norms[None, :]
    sft_prod = sft_prod.reshape(shape + (sft_prod.shape[-1],))

    return np.flip(sft_prod[..., :T], axis=-1), sft_prod[..., T - 1:]


def apply_exp_kernel(
    x: np.ndarray, 
    beta: float | None, 
    ksize: int
) -> np.ndarray:
    """ Apply an exponential 1d kernel to last dimension of x.

    :param x: (B) x T array
    :param beta: the exponential average parameter
    :param ksize: the size of the kernel, will cut the kernel tail 
    """
    if beta is None:
        return x
    
    if beta ** (-1) > x.shape[-1] / 4:
        raise ValueError("beta is too small, the kernel would be too large") 

    shape = x.shape
    x = x.reshape((-1, shape[-1]))

    # exponential kernel
    kernel = np.exp(-beta*np.arange(x.shape[-1])) 
    ksize = ksize or x.shape[-1]
    if ksize is not None:
        kernel[ksize:] = 0
    kernel /= kernel.sum()

    # padding
    kernel = np.pad(kernel, (0, ksize), 'constant')
    x = np.pad(x, ((0, 0), (ksize, 0)), 'constant')

    # convolution
    x_hat = np.fft.fft(x)
    kernel_hat = np.fft.fft(kernel[None, :])
    x_avg = np.fft.ifft(x_hat * kernel_hat).real

    return x_avg[:, ksize:].reshape(shape)


def shifted_product(
    x: np.ndarray, 
    y: np.ndarray, 
    beta: float | None, 
    ksize: int
) -> tuple[np.ndarray, ...]:
    """
    Computes < x_tilde(t-tau) y(t) >_t where x_tilde is an exponential 
    average of x in the past if tau => 0, in the future otherwise.

    :param x: ... x T array
    :param y: ... x T array
    :param beta: the exponential average parameter
    :param ksize: the size of the exponential kernel
    :return:
    """
    x_pos = apply_exp_kernel(x, beta, ksize)  # x is smoothed in the past
    x_neg = np.flip(apply_exp_kernel(np.flip(x, axis=-1), beta, ksize), axis=-1)  # x is smoothed in the future

    pos, _ = shifted_product_aux(x_pos, y)
    _, neg = shifted_product_aux(x_neg, y)

    tau_pos = np.arange(x.shape[-1])

    return tau_pos, -tau_pos, pos, neg


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, axis: int = -1) -> np.ndarray:
    """ Identical to scipy r2_score but specifying axis. """
    def MSE(a, b):
        return ((a - b) ** 2).mean(axis)
    return 1 - MSE(y_true, y_pred) / MSE(y_true, y_true.mean(axis, keepdims=True))


@contextlib.contextmanager
def set_seed(seed: int | None) -> Generator:
    if seed is not None:
        saved_state = np.random.get_state()
        # Set the new seed
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(saved_state)
    else:
        # If seed is None, do nothing special
        yield


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    return tensor.detach().numpy()
