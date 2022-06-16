import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

import utils.complex_utils as cplx
from utils import to_numpy
from data_source import FBmLoader, PoissonLoader, MRWLoader, SMRWLoader
from scattering_network.scattering import Scattering
from scattering_network.moments import Marginal, Cov, CovStat
from scattering_network.module_chunk import ModuleChunk


##################
# DATA LOADING
##################

def load_data(name, R, T, cache_dir=None, **data_param):
    """Time series data loading function.

    :param name: fbm, poisson, mrw, smrw, hawkes, turbulence or snp
    :param R: number of realizations
    :param T: number of time samples
    :param cache_dir: the directory used to cache trajectories
    :return: dataloader
    """
    if name == 'snp':
        raise ValueError("S&P data is private, please provide your own S&P data")
    if name == 'heliumjet':
        raise ValueError("S&P data is private, please provide your own S&P data")

    if cache_dir is None:
        cache_dir = Path(os.getcwd()) / 'cached_dir'

    loader = {
        'fbm': FBmLoader,
        'poisson': PoissonLoader,
        'mrw': MRWLoader,
        'smrw': SMRWLoader,
        # 'hawkes': HawkesLoader,
    }

    dtld = loader[name](cache_dir)
    X = dtld.load(R=R, T=T, **data_param).X

    return X[:, 0, 0, :]


##################
# ANALYSIS
##################

def init_model(N, T, J, Q, r_max, A, wav_type, high_freq, rm_high, wav_norm, normalize,
               moments, m_types, chunk_method, nchunks):
    """Initialize a scattering covariance model.

    :param N: number of in_data channel
    :param T: number of time samples
    :param Q: wavelets per octave
    :param r_max: number convolution layers
    :param A: number of angles
    :param wav_type: wavelet type
    :param high_freq: central frequency of mother wavelet
    :param rm_high: wether to remove high frequencies from data before scattering
    :param wav_norm: wavelet normalization i.e. l1, l2
    :param normalize: wether to have a normalization layer after first convolution
    :param moments: moments to compute on scattering i.e. marginal, cov, covstat
    :param m_types: the type of moments to compute i.e. m00, m10, m11
    :param chunk_method: the method to optimize the coefficient graph i.e. quotient_n, graph_optim
    :param nchunks: the number of chunks

    :return: a torch module
    """
    module_list = [Scattering(J, Q, r_max, T, A, N, wav_type, high_freq, rm_high, wav_norm, normalize)]

    if moments == 'marginal':
        module_list.append(Marginal(qs=[2.0]))
    if moments in ['cov', 'covstat']:
        module_list.append(Cov(J, Q, r_max, N, m_types))
    if moments == 'covstat':
        module_list.append(CovStat(J * Q, m_types))

    return ModuleChunk(module_list, chunk_method, nchunks)


def compute_scattering_normalization(xf_torch, J, Q, wav_type, high_freq, wav_norm):
    """Compute sigma^2(j).
    :param xf_torch: a 1 x N x T x 2 tensor
    :param J: number of octaves
    :param Q: number of scales per octave
    :param wav_type: wavelet type
    :param high_freq: central frequency of mother wavelet
    :param wav_norm: wavelet normalization i.e. l1, l2

    :return: a tensor
    """
    # TODO. Should be implemented as a proper normalization layer
    model_avg = init_model(
        N=xf_torch.shape[1], T=xf_torch.shape[2], J=J, Q=Q, r_max=1, A=None,
        wav_type=wav_type, high_freq=high_freq, rm_high=True, wav_norm=wav_norm, normalize=False,
        moments='marginal', m_types=['m00'], chunk_method='quotient_n', nchunks=xf_torch.shape[1]
    )
    model_avg.clear_params()
    model_avg.init_chunks()
    model_avg.cuda()  # TODO. Implement cpu possibility
    return cplx.real(model_avg(xf_torch).mean('n').select(q=2.0)).pow(0.5)


def analyze(X, J=None, Q=1, wav_type='battle_lemarie', wav_norm='l1', high_freq=0.425,
            rm_high=False, moments='covstat', m_types=['m00', 'm10', 'm11'], normalize=False, nchunks=None,
            cuda=False):
    """Compute sigma^2(j).
    :param X: a R x T array
    :param J: number of octaves
    :param Q: number of scales per octave
    :param wav_type: wavelet type
    :param wav_norm: wavelet normalization i.e. l1, l2
    :param high_freq: central frequency of mother wavelet
    :param rm_high: preprocess data by doing a high pass filter
    :param moments: the type of moments to compute
    :param m_types: m00: sigma^2 and s^2, m10: cp, m11: cm
    :param normalize: normalize, wether first wavelet layer
    :param nchunks: nb of chunks, increase it to reduce memory usage

    :return: a ModelOutput result
    """
    R, T = X.shape

    if J is None:
        J = int(np.log2(T)) - 3

    X = cplx.from_np(X).unsqueeze(0)

    # initialize model
    model = init_model(N=R, T=T, J=J, Q=Q, r_max=2, A=None, wav_type=wav_type, high_freq=high_freq, rm_high=rm_high,
                       wav_norm=wav_norm, normalize=normalize, moments=moments, m_types=m_types,
                       chunk_method='quotient_n', nchunks=nchunks or R)
    model.init_chunks()

    # compute
    if cuda:
        X = X.cuda()
        model = model.cuda()

    RX = model(X)

    return RX.cpu()


##################
# SYNTHESIS
##################


##################
# VIZUALIZE
##################

def plot_spectrum_bis(RXs, labels=None, linewidth=3.0):
    if len(RXs) != len(labels):
        return ValueError("Invalid number of labels")

    n_outputs = len(RXs)
    labels = labels or [''] * n_outputs

    extract_moments = lambda RX, q: cplx.real(RX.select(pivot='n1', m_type='m00', q=2, lp=False)).log2().mean(0)

    # coefficients sigma^2(j)
    log_spectrums = [extract_moments(RX, 2) for RX in RXs]

    # coefficients sparsity^2(j)
    log_sparsity = [2 * extract_moments(RX, 1) - extract_moments(RX, 2) for RX in RXs]

    # plot
    js = np.arange(log_spectrums[0].size)
    plt.figure(figsize=(10, 10))
    for i_lb, (lb, RX) in enumerate(zip(labels, RXs)):
        plt.subplot2grid((1, 2), (0, 0))
        plt.plot(-js, log_spectrums, label=lb, linewidth=linewidth)
        plt.scatter(-js, log_spectrums, marker='+', s=200, linewidth=linewidth)
        plt.subplot2grid((1, 2), (0, 1))
        plt.plot(-js, log_sparsity, label=lb, linewidth=linewidth)
        plt.scatter(-js, log_sparsity, marker='+', s=200, linewidth=linewidth)


def plot_marginal_moments(RXs, labels=None, linewidth=3.0, fontsize=30, offset=0, errorbars=False):
    if labels is not None and len(RXs) != len(labels):
        return ValueError("Invalid number of labels")

    labels = labels or [''] * len(RXs)
    labels_here = labels
    J = RXs[0].idx_info.j.max()

    Cmarginal = np.zeros((len(labels), 3, J))
    estim_error = np.zeros((len(labels), 3, J))
    for i_lb, (label, RX) in enumerate(zip(labels, RXs)):
        WX1 = RX.select(pivot='n1', m_type='m00', q=1, lp=False)
        WX2 = RX.select(pivot='n1', m_type='m00', q=2, lp=False)

        m1_nj = torch.log2(cplx.real(WX1))
        m2_nj = torch.log2(cplx.real(WX2))
        ms_nj = 2 * m1_nj - m2_nj
        if errorbars:
            # marginal moments
            Cmarginal[i_lb, 0, :] = m1_nj.mean(0)
            Cmarginal[i_lb, 1, :] = m2_nj.mean(0)
            Cmarginal[i_lb, 2, :] = ms_nj.mean(0)

            N = m1_nj.shape[0]
            estim_error[i_lb, 0, :] = (m1_nj - m1_nj.mean(0, keepdim=True)).pow(2.0).sum(0).div(N - 1).div(N).pow(0.5)
            estim_error[i_lb, 1, :] = (m2_nj - m1_nj.mean(0, keepdim=True)).pow(2.0).sum(0).div(N - 1).div(N).pow(0.5)
            estim_error[i_lb, 2, :] = (ms_nj - m1_nj.mean(0, keepdim=True)).pow(2.0).sum(0).div(N - 1).div(N).pow(0.5)
        else:
            Cmarginal[i_lb, 0, :] = 2 ** (m1_nj.mean(0))
            Cmarginal[i_lb, 1, :] = 2 ** (m2_nj.mean(0))
            Cmarginal[i_lb, 2, :] = 2 ** (ms_nj.mean(0))

    if errorbars:
        Cmarginal += 2 * np.arange(len(labels))[:, None, None]

    else:
        Cmarginal[:, :2, :] /= Cmarginal[:, :2, -1:]

    # PLOT 1 sigma^2
    plt.figure(figsize=(20, 10))
    plt.subplot2grid((1, 2), (0, 0))
    js = np.arange(offset, Cmarginal.shape[-1])
    for i_lb, lb in enumerate(labels_here):
        if errorbars:
            plt.errorbar(-js, Cmarginal[i_lb, 1, offset:], yerr=estim_error[i_lb, 1, offset:], label=lb, capsize=4)
        else:
            plt.plot(-js, Cmarginal[i_lb, 1, offset:], label=lb, linewidth=linewidth)
            plt.scatter(-js, Cmarginal[i_lb, 1, offset:], marker='+', s=200, linewidth=linewidth)
    if not errorbars:
        plt.yscale('log', base=2)
    plt.xlabel(r'$-j$', fontsize=fontsize + 30)
    plt.yticks(fontsize=fontsize)
    plt.xticks(-js, [fr'$-{j + 1}$' for j in js], fontsize=fontsize)

    # PLOT 2 sparsity^2
    plt.subplot2grid((1, 2), (0, 1))
    for i_lb, lb in enumerate(labels_here):
        if errorbars:
            plt.errorbar(-js, Cmarginal[i_lb, 2, offset:], yerr=estim_error[i_lb, 2, offset:], label=lb, capsize=4)
        else:
            plt.plot(-js, Cmarginal[i_lb, 2, offset:], label=lb, linewidth=linewidth)
            plt.scatter(-js, Cmarginal[i_lb, 2, offset:], marker='+', s=200, linewidth=linewidth)
    if not errorbars:
        plt.yscale('log', base=2)
        plt.ylim(2 ** (-4), 1.0)
    plt.xlabel(r'$-j$', fontsize=fontsize + 30)
    plt.yticks(fontsize=fontsize)
    plt.xticks(-js, [fr'$-{j + 1}$' for j in js], fontsize=fontsize)


def plot_cross_phased(RXs, labels=None, linewidth=3.0, fontsize=30, offset=0, error_bars=False, estim_bars=False,
                      theta_threshold=0.0075, single_plot=False):
    if labels is not None and len(RXs) != len(labels):
        return ValueError("Invalid number of labels")

    labels_here = labels
    labels_curve = ['']
    J = RXs[0].idx_info.j.max()
    cOlors = ['skyblue', 'coral', 'lightgreen', 'darkgoldenrod', 'mediumpurple', 'red', 'purple', 'black',
              'paleturquoise'] + ['orchid'] * 20

    cp = torch.zeros(len(labels), J - 1, 2)
    err_mod = torch.zeros(len(labels), J - 1)
    err_estim = torch.zeros(len(labels), J - 1)

    def error_arg(mod, err_mod):
        return np.arctan(err_mod / mod / np.sqrt(np.clip(1 - err_mod ** 2 / mod ** 2, 1e-6, 1)))

    for i_lb, (RX, lb, color) in enumerate(zip(RXs, labels_here, cOlors)):
        N = np.unique(RX.idx_info['n1'].values).size
        norm2 = RX.select(pivot='n1', m_type='m00', q=2, lp=False)[:, :, 0].unsqueeze(-1)

        for alpha in range(1, J):
            cp_nj = torch.zeros(N, J - alpha, 2)
            for j1 in range(alpha, J):
                coeff = RX.select(pivot='n1', m_type='m10', j1=j1 - alpha, jp1=j1, lp=False)[:, 0, :]
                coeff /= norm2[:, j1, ...].pow(0.5) * norm2[:, j1 - alpha, ...].pow(0.5)
                cp_nj[:, j1 - alpha, :] = coeff

            # the mean in j of the variance of time estimators
            cp[i_lb, alpha - 1, :] = cp_nj.mean(0).mean(0)
            cp_err_j = (cplx.modulus(cp_nj).pow(2.0).mean(0) - cplx.modulus(cp_nj.mean(0)).pow(2.0)) / N
            err_mod[i_lb, alpha - 1] = cp_err_j.mean(0).pow(0.5)
            if not error_bars:
                err_mod[i_lb, alpha - 1] = 0.0
                err_estim[i_lb, alpha - 1] = 0.0

    cp, cp_mod, cp_arg = cplx.to_np(cp), np.abs(cplx.to_np(cp)), np.angle(cplx.to_np(cp))
    err_mod, err_estim, err_arg = to_numpy(err_mod), to_numpy(err_estim), error_arg(cp_mod, to_numpy(err_mod))

    # phase instability at z=0
    cp_arg[cp_mod < theta_threshold] = 0.0
    err_arg[cp_mod < theta_threshold] = 0.0

    cp_arg = np.abs(cp_arg)

    def plot_modulus(i_lb, lb, lb_curve, ax, color, y, y_err, y_err_estim):
        alphas = np.arange(1, J)
        if error_bars:
            plt.errorbar(alphas, y, yerr=y_err, capsize=4, color=color,
                         linestyle=(0, (5, 5)) if 'standard' in lb_curve else '-', label=lb_curve)
            if estim_bars:
                eb = plt.errorbar(alphas + 0.05, y, yerr=y_err_estim, capsize=4, color=color, fmt=' ')
                eb[-1][0].set_linestyle('--')
        else:
            plt.plot(alphas, y, color=color or 'green', label=lb_curve)
            plt.scatter(alphas, y, color=color or 'green', marker='+')
        plt.title(lb, fontsize=fontsize)
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.ylim(-0.02, 0.09)
        if i_lb == 0:
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.ylabel(r'$\mathrm{Modulus}$', fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
        else:
            plt.tick_params(labelleft=False)
        plt.tick_params(labelbottom=False)
        plt.locator_params(axis='y', nbins=5)

    def plot_phase(i_lb, lb, lb_curve, color, y, y_err):
        alphas = np.arange(1, J)
        plt.plot(alphas, y, color=color, linestyle=(0, (5, 5)) if 'standard' in lb_curve else '-', label=lb_curve)
        plt.scatter(alphas, y, color=color, marker='+')
        plt.xticks(np.arange(1, J), [(rf'${j}$' if j % 2 == 1 else '') for j in np.arange(1, J)], fontsize=fontsize)
        plt.yticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi],
                   [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=fontsize)
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xlabel(r'$a$', fontsize=fontsize + 10)
        if i_lb == 0:
            plt.ylabel(r'$\mathrm{Phase}$', fontsize=fontsize)
        else:
            plt.tick_params(labelleft=False)

    plt.figure(figsize=(5, 10) if single_plot else (len(labels) * 5, 10))
    for i_lb, (lb, lb_curve) in enumerate(zip(labels_here, labels_curve)):
        # ax_mod = plt.subplot2grid((2, len(labels)), (0, i_lb))
        #     if i_lb == 0:
        #         ax_mod.yaxis.set_tick_params(which='major', direction='in', width=1.5, length=7)
        plot_modulus(i_lb, lb, lb_curve, None, cOlors[i_lb] if single_plot else 'black', cp_mod[i_lb], err_mod[i_lb],
                     err_estim[i_lb])
        # tx = ax_mod.yaxis.get_offset_text().set_fontsize(fontsize - 20)

    for i_lb, (lb, lb_curve) in enumerate(zip(labels_here, labels_curve)):
        # if i_lb == 0:  # define new plot
        #     ax_ph = plt.subplot2grid((2, np.unique(i_graphs).size), (1, i_graphs[i_lb]))
        plot_phase(i_lb, lb, lb_curve, cOlors[i_lb] if single_plot else 'black', cp_arg[i_lb], None)

    plt.tight_layout()

