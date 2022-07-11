import os
from pathlib import Path
from itertools import product
import numpy as np
import torch
import matplotlib.pyplot as plt

import utils.complex_utils as cplx
from utils import to_numpy
from data_source import FBmLoader, PoissonLoader, MRWLoader, SMRWLoader
from scattering_network.scale_indexer import ScaleIndexer
from scattering_network.time_layers import Wavelet
from scattering_network.moments import Marginal, Cov, CovStat
from scattering_network.module_chunk import ModuleChunk, SkipConnection
from scattering_network.described_tensor import DescribedTensor


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

def init_model(N, T, J, Q1, Q2, r_max, wav_type, high_freq, rm_high, wav_norm, normalize,
               moments, m_types, chunk_method, nchunks):
    """Initialize a scattering covariance model.

    :param N: number of in_data channel
    :param T: number of time samples
    :param Q1: wavelets per octave first layer
    :param Q2: wavelets per octave second layer
    :param r_max: number convolution layers
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
    module_list = []

    # scattering network
    sc_idxer = ScaleIndexer(J, Q1, Q2, r_max)
    W1 = Wavelet(T, J, Q1, wav_type, high_freq, wav_norm, 1, sc_idxer)
    module_list.append(W1)

    if r_max > 1:
        W2 = Wavelet(T, J, Q2, wav_type, high_freq, wav_norm, 2, sc_idxer)
        module_list.append(SkipConnection(W2))

    # moments
    if moments == 'marginal':
        module_list.append(Marginal(qs=[1.0, 2.0]))
    if moments in ['cov', 'covstat']:
        module_list.append(Cov(N, sc_idxer, m_types))
    if moments == 'covstat':
        module_list.append(CovStat(J * Q1, m_types))

    model = ModuleChunk(module_list, chunk_method, N, nchunks)

    model.clear_params()
    model.init_chunks()

    return model


def compute_scattering_normalization(xf_torch, J, Q1, Q2, wav_type, high_freq, wav_norm):
    """Compute sigma^2(j).
    :param xf_torch: a 1 x N x T x 2 tensor
    :param J: number of octaves
    :param Q1: number of scales per octave first layer
    :param Q2: number of scales per octave second layer
    :param wav_type: wavelet type
    :param high_freq: central frequency of mother wavelet
    :param wav_norm: wavelet normalization i.e. l1, l2

    :return: a tensor
    """
    # TODO. Should be implemented as a proper normalization layer
    model_avg = init_model(
        N=xf_torch.shape[1], T=xf_torch.shape[2], J=J, Q1=Q1, Q2=Q2, r_max=1,
        wav_type=wav_type, high_freq=high_freq, rm_high=False, wav_norm=wav_norm, normalize=False,
        moments='marginal', m_types=['m00'], chunk_method='quotient_n', nchunks=xf_torch.shape[1]
    )
    model_avg.clear_params()
    model_avg.init_chunks()
    model_avg.cuda()  # TODO. Implement cpu possibility
    return cplx.real(model_avg(xf_torch).mean('n1').select(q=2.0)).pow(0.5)


def analyze(X, J=None, Q1=1, Q2=1, wav_type='battle_lemarie', wav_norm='l1', high_freq=0.425,
            rm_high=False, moments='covstat', m_types=None, keep_past=False,
            nchunks=None, cuda=False):
    """Compute sigma^2(j).
    :param X: a R x T array
    :param J: number of octaves
    :param Q1: number of scales per octave on first wavelet layer
    :param Q2: number of scales per octave on second wavelet layer
    :param wav_type: wavelet type
    :param wav_norm: wavelet normalization i.e. l1, l2
    :param high_freq: central frequency of mother wavelet
    :param rm_high: preprocess data by doing a low pass filter
    :param moments: the type of moments to compute
    :param m_types: m00: sigma^2 and s^2, m10: cp, m11: cm
    :param nchunks: nb of chunks, increase it to reduce memory usage

    :return: a ModelOutput result
    """
    R, T = X.shape

    if J is None:
        J = int(np.log2(T)) - 3

    X = cplx.from_np(X).unsqueeze(0)

    # initialize model
    model = init_model(N=R, T=T, J=J, Q1=Q1, Q2=Q2, r_max=2, wav_type=wav_type, high_freq=high_freq, rm_high=rm_high,
                       wav_norm=wav_norm, normalize=moments == 'covstat', moments=moments,
                       m_types=m_types or ['m00', 'm10', 'm11'],
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

COLORS = ['skyblue', 'coral', 'lightgreen', 'darkgoldenrod', 'mediumpurple', 'red', 'purple', 'black',
          'paleturquoise'] + ['orchid'] * 20


def bootstrap_variance_complex(X, n_points, n_samples):
    """Estimate variance of tensor X along last axis using bootstrap method."""
    # sample data uniformly
    sampling_idx = np.random.randint(low=0, high=X.shape[-2], size=(n_samples, n_points))
    sampled_data = X[..., sampling_idx, :]

    # computes mean
    mean = sampled_data.mean(-2).mean(-2)

    # computes bootstrap variance
    var = (cplx.modulus(sampled_data.mean(-2) - mean[..., None, :]).pow(2.0)).sum(-1) / (n_samples - 1)

    return mean, var


def error_arg(mod, err_mod):
    return np.arctan(err_mod / mod / np.sqrt(np.clip(1 - err_mod ** 2 / mod ** 2, 1e-6, 1)))


def plot_marginal_moments(RXs, estim_err=False,
                          axes=None, labels=None, linewidth=3.0, fontsize=30):
    """
    Plot the marginal moments
        - (wavelet power spectrum) sigma^2(j)
        - (sparsity factors) s^2(j)

    :param RXs: ModelOutput or list of ModelOutput
    :param estim_err: display estimation error due to estimation on several realizations
    :param axes: custom axes: array of size 2
    :param labels: list of labels for each model output
    :param linewidth: curve linewidth
    :param fontsize: labels fontsize
    :return:
    """
    if isinstance(RXs, DescribedTensor):
        RXs = [RXs]
    if labels is not None and len(RXs) != len(labels):
        raise ValueError("Invalid number of labels")
    if estim_err:
        raise ValueError("Estim error plot not yet supported.")
    if axes is not None and axes.size != 2:
        raise ValueError("The axes provied to plot_marginal_moments should be an array of size 2.")

    labels = labels or [''] * len(RXs)
    axes = None if axes is None else axes.ravel()

    def get_data(RX, q):
        WX_nj = cplx.real(RX.select(pivot='n1', r=1, m_type='m00', q=q, low=False))
        logWX_nj = torch.log2(WX_nj)
        return logWX_nj

    def get_variance(WX_nj):
        N = WX_nj.shape[0]
        return (WX_nj - WX_nj.mean(0, keepdim=True)).pow(2.0).sum(0).div(N-1).div(N)

    def plot_exponent(js, i_ax, lb, y, y_err):
        if axes is None:
            plt.subplot2grid((1, 2), (0, i_ax))
        else:
            plt.sca(axes[i_ax])
        if estim_err:
            plt.errorbar(-js, y, yerr=y_err, label=lb, capsize=4)
        else:
            plt.plot(-js, y, label=lb, linewidth=linewidth)
            plt.scatter(-js, y, marker='+', s=200, linewidth=linewidth)
        if not estim_err or True:
            plt.yscale('log', base=2)
        ax = plt.gca()
        a, b = ax.get_ylim()
        if i_ax == 0:
            ax.set_ylim(min(a, 2**(-2)), max(b, 2**2))
        else:
            ax.set_ylim(min(2**(-3), a), 1.0)
            # plt.ylim(2 ** (-4), 1.0)
        plt.xlabel(r'$-j$', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(-js, [fr'$-{j + 1}$' for j in js], fontsize=fontsize)
        plt.legend(prop={'size': 15})

    for i_lb, (lb, RX) in enumerate(zip(labels, RXs)):
        if 'm_type' not in RX.idx_info.columns:
            raise ValueError("The model output does not have the moments ")
        js = np.unique(RX.idx_info.reduce(low=False).j1)

        has_power_spectrum = 2.0 in RX.idx_info.q.values
        has_sparsity = 1.0 in RX.idx_info.q.values

        if has_power_spectrum:
            logWX2_n = get_data(RX, 2.0)
            logWX2_err = get_variance(logWX2_n) ** 0.5
            logWX2 = logWX2_n.mean(0)
            logWX2 -= logWX2[-1].item()
            plot_exponent(js, 0, lb, 2.0 ** logWX2, 2.0 ** logWX2_err)
            plt.title('Wavelet Spectrum', fontsize=fontsize)

            if has_sparsity:
                logWX1_n = get_data(RX, 1.0)
                logWXs_n = 2 * logWX1_n - logWX2_n
                logWXs_err = get_variance(logWXs_n) ** 0.5
                logWXs = logWXs_n.mean(0)
                plot_exponent(js, 1, lb, 2.0 ** logWXs, 2.0 ** logWXs_err)
                plt.title('Sparsity factor', fontsize=fontsize)


def plot_cross_phased(RXs, estim_err=False, self_simi_err=False, theta_threshold=0.0075,
                      axes=None, labels=None, fontsize=30, single_plot=False, ylim=0.09):
    """
    Plot the phase-envelope cross-spectrum C_{W|W|}(a) as two graphs : |C_{W|W|}| and Arg(C_{W|W|}).

    :param RXs: ModelOutput or list of ModelOutput
    :param estim_err: display estimation error due to estimation on several realizations
    :param self_simi_err: display self-similarity error, it is a measure of scale regularity
    :param theta_threshold: rules phase instability
    :param axes: custom axes: array of size 2
    :param labels: list of labels for each model output
    :param fontsize: labels fontsize
    :param single_plot: output all ModelOutput on a single plot
    :param ylim: above y limit of modulus graph
    :return:
    """
    if isinstance(RXs, DescribedTensor):
        RXs = [RXs]
    if labels is not None and len(RXs) != len(labels):
        raise ValueError("Invalid number of labels")
    if estim_err:
        raise ValueError("Estim error plot not yet supported.")

    labels = labels or [''] * len(RXs)
    labels_here = labels
    labels_curve = labels
    columns = RXs[0].idx_info.columns
    J = RXs[0].idx_info.j.max() if 'j' in columns else RXs[0].idx_info.j1.max()

    cp = torch.zeros(len(labels), J - 1, 2)
    err_mod = torch.zeros(len(labels), J - 1)
    err_estim = torch.zeros(len(labels), J - 1)

    def error_arg(mod, err_mod):
        return np.arctan(err_mod / mod / np.sqrt(np.clip(1 - err_mod ** 2 / mod ** 2, 1e-6, 1)))

    for i_lb, (RX, lb, color) in enumerate(zip(RXs, labels_here, COLORS)):
        N = np.unique(RX.idx_info['n1'].values).size
        norm2 = RX.select(pivot='n1', m_type='m00', q=2, low=False)[:, :, 0].unsqueeze(-1)

        for alpha in range(1, J):
            cp_nj = torch.zeros(N, J - alpha, 2)
            for j1 in range(alpha, J):
                coeff = RX.select(pivot='n1', m_type='m10', j1=j1 - alpha, jp1=j1, low=False)[:, 0, :]
                coeff /= norm2[:, j1, ...].pow(0.5) * norm2[:, j1 - alpha, ...].pow(0.5)
                cp_nj[:, j1 - alpha, :] = coeff

            # the mean in j of the variance of time estimators
            cp[i_lb, alpha - 1, :] = cp_nj.mean(0).mean(0)
            cp_err_j = (cplx.modulus(cp_nj).pow(2.0).mean(0) - cplx.modulus(cp_nj.mean(0)).pow(2.0)) / N
            err_mod[i_lb, alpha - 1] = cp_err_j.mean(0).pow(0.5)
            if not self_simi_err:
                err_mod[i_lb, alpha - 1] = 0.0
                err_estim[i_lb, alpha - 1] = 0.0

    cp, cp_mod, cp_arg = cplx.to_np(cp), np.abs(cplx.to_np(cp)), np.angle(cplx.to_np(cp))
    err_mod, err_estim, err_arg = to_numpy(err_mod), to_numpy(err_estim), error_arg(cp_mod, to_numpy(err_mod))

    # phase instability at z=0
    cp_arg[cp_mod < theta_threshold] = 0.0
    err_arg[cp_mod < theta_threshold] = 0.0

    cp_arg = np.abs(cp_arg)

    def plot_modulus(i_lb, lb, lb_curve, color, y, y_err, y_err_estim):
        alphas = np.arange(1, J)
        if self_simi_err:
            plt.errorbar(alphas, y, yerr=y_err, capsize=4, color=color,
                         linestyle=(0, (5, 5)) if 'standard' in lb_curve else '-', label=lb_curve)
            if estim_err:
                eb = plt.errorbar(alphas + 0.05, y, yerr=y_err_estim, capsize=4, color=color, fmt=' ')
                eb[-1][0].set_linestyle('--')
        else:
            plt.plot(alphas, y, color=color or 'green', label=lb_curve)
            plt.scatter(alphas, y, color=color or 'green', marker='+')
        plt.title("Phase-Env spectrum \n (Modulus)", fontsize=fontsize)
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xticks(np.arange(1, J),
                   [(rf'${j}$' if j % 2 == 1 else '') for j in np.arange(1, J)], fontsize=fontsize)
        plt.xlabel(r'$a$', fontsize=fontsize)
        plt.ylim(-0.02, ylim)
        if i_lb == 0:
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.yticks(fontsize=fontsize)
        plt.locator_params(axis='y', nbins=5)
        plt.legend(prop={'size': 15})

    def plot_phase(lb_curve, color, y):
        alphas = np.arange(1, J)
        plt.plot(alphas, y, color=color, linestyle=(0, (5, 5)) if 'standard' in lb_curve else '-',
                 label=lb_curve)
        plt.scatter(alphas, y, color=color, marker='+')
        plt.xticks(np.arange(1, J), [(rf'${j}$' if j % 2 == 1 else '') for j in np.arange(1, J)], fontsize=fontsize)
        plt.yticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi],
                   [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=fontsize)
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xlabel(r'$a$', fontsize=fontsize)
        plt.title("Phase-Env spectrum \n (Phase)", fontsize=fontsize)
        plt.legend(prop={'size': 15})

    if axes is None:
        plt.figure(figsize=(5, 10) if single_plot else (len(labels) * 5, 10))
        ax_mod = plt.subplot2grid((2, 1), (0, 0))
        ax_mod.yaxis.set_tick_params(which='major', direction='in', width=1.5, length=7)
    else:
        plt.sca(axes[0])
    for i_lb, (lb, lb_curve) in enumerate(zip(labels_here, labels_curve)):
        plot_modulus(i_lb, lb, lb_curve, COLORS[i_lb], cp_mod[i_lb], err_mod[i_lb], err_estim[i_lb])

    if axes is None:
        plt.subplot2grid((2, 1), (1, 0))
    else:
        plt.sca(axes[1])
    for i_lb, (lb, lb_curve) in enumerate(zip(labels_here, labels_curve)):
        plot_phase(lb_curve, COLORS[i_lb], cp_arg[i_lb])
    if axes is None:
        plt.tight_layout()


def plot_modulus(RXs, estim_err=False, self_simi_err=False, bootstrap=True, theta_threshold=0.0075,
                 axes=None, labels=None, fontsize=40, ylim=3.5):
    """
    Plot the scattering cross-spectrum C_S(a,b) as two graphs : |C_S| and Arg(C_S).

    :param RXs: ModelOutput or list of ModelOutput
    :param estim_err: display estimation error due to estimation on several realizations
    :param self_simi_err: display self-similarity error, it is a measure of scale regularity
    :param theta_threshold: rules phase instability
    :param axes: custom axes: array of size 2 x labels
    :param labels: list of labels for each model output
    :param fontsize: labels fontsize
    :param ylim: above y limit of modulus graph
    :return:
    """
    if isinstance(RXs, DescribedTensor):
        RXs = [RXs]
    if labels is not None and len(RXs) != len(labels):
        raise ValueError("Invalid number of labels.")
    if estim_err:
        raise ValueError("Estim error plot not yet supported.")
    if axes is not None and axes.size != 2 * len(RXs):
        raise ValueError(f"Existing axes must be provided as an array of size {2 * len(RXs)}")

    axes = None if axes is None else axes.reshape(2, len(RXs))

    # infer model type
    if 'j1' in RXs[0].idx_info:
        model_type = 'cov'
    elif 'j' in RXs[0].idx_info:
        model_type = 'covstat'
    else:
        raise ValueError("Unrecognized model type.")

    if self_simi_err and model_type == 'covstat':
        raise ValueError("Impossible to output self-similarity error on covstat model. Use a cov model instead.")

    labels = labels or [''] * len(RXs)
    i_graphs = np.arange(len(labels))
    columns = RXs[0].idx_info.columns
    J = RXs[0].idx_info.j.max() if 'j' in columns else RXs[0].idx_info.j1.max()
    N = RXs[0].idx_info.n1.max() + 1

    C_env = torch.zeros(len(labels), J - 1, J - 1, 2)
    model_bias = torch.zeros(len(labels), J - 1, J - 1)
    estim_err = torch.zeros(len(labels), J - 1, J - 1)

    for i_lb, (RX, lb, color) in enumerate(zip(RXs, labels, COLORS)):
        norm2 = RX.select(pivot='n1', m_type='m00', q=2, low=False)[:, :, 0].unsqueeze(-1)  # N x J x 1

        for (a, b) in product(range(J - 1), range(-J + 1, 0)):
            if a - b >= J:
                continue

            # prepare covariances
            if model_type == 'cov':
                C_env_nj = torch.zeros(N, J + b - a, 2)
                for j1 in range(a, J + b):
                    coeff = RX.select(pivot='n1', m_type='m11', j1=j1, jp1=j1 - a, j2=j1 - b, low=False)[:, 0, :]
                    coeff /= norm2[:, j1, ...].pow(0.5) * norm2[:, j1 - a, ...].pow(0.5)
                    C_env_nj[:, j1 - a, :] = coeff

                C_env_j = C_env_nj.mean(0)
                C_env[i_lb, a, J - 1 + b, :] = C_env_j.mean(0)
                if b == -J + a + 1:
                    model_bias[i_lb, a, J - 1 + b] = 0.0
                else:
                    model_bias[i_lb, a, J - 1 + b] = cplx.modulus(C_env_j - C_env_j.mean(0, keepdim=True)) \
                        .pow(2.0).sum(0).div(J + b - a - 1).pow(0.5)
                # compute estimation error
                if bootstrap:
                    mean, var = bootstrap_variance_complex(C_env_nj.transpose(0, 1), C_env_nj.shape[0], 20000)
                    estim_err[i_lb, a, J - 1 + b] = var.mean(0).pow(0.5)
                else:
                    estim_err[i_lb, a, J - 1 + b] = (cplx.modulus(C_env_nj).pow(2.0).mean(0) -
                                                     cplx.modulus(C_env_nj.mean(0)).pow(2.0)) / (N - 1)
            else:
                coeff_ab = RX.select(pivot='n1', m_type='m11', alpha=a, beta=b, low=False)[:, 0, :]
                C_env[i_lb, a, J - 1 + b, :] = coeff_ab.mean(0)

    C_env, C_env_mod = cplx.to_np(C_env), np.abs(cplx.to_np(C_env))
    model_bias, estim_err = to_numpy(model_bias), to_numpy(estim_err)

    # power spectrum normalization
    betas = np.arange(-J + 1, 0)[None, :]
    C_env_mod /= (2.0 ** betas)
    model_bias /= (2.0 ** betas)
    estim_err /= (2.0 ** betas)

    # phase
    C_env_arg = np.angle(C_env)
    C_env_arg[C_env < theta_threshold] = 0.0

    def plot_modulus(i_lb, lb, y, y_err, y_err_estim):
        for alpha in range(J - 1):
            betas = np.arange(-J + 1 + alpha, 0)
            if self_simi_err:
                plt.errorbar(betas, y[alpha, alpha:], yerr=y_err[alpha, alpha:], capsize=4,
                            label=lb if alpha == 0 else '')
                if self_simi_err:
                    eb = plt.errorbar(betas + 0.05, y[alpha, alpha:], yerr=y_err_estim[alpha, alpha:],
                                      capsize=4, fmt=' ')
                    eb[-1][0].set_linestyle('--')
            else:
                plt.plot(betas, y[alpha, alpha:], label=lb if alpha == 0 else '')
                plt.scatter(betas, y[alpha, alpha:], marker='+')
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xticks(np.arange(-J + 1, 0), [(rf'${beta}$' if beta % 2 == 1 else '') for beta in np.arange(-J + 1, 0)],
                   fontsize=fontsize)
        plt.xlabel(r'$b$', fontsize=fontsize)
        plt.ylim(-0.02, ylim)
        if i_lb == 0:
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#             plt.ylabel(r'$\mathrm{Modulus}$', fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
#         else:
#             plt.tick_params(labelleft=False)
        plt.locator_params(axis='x', nbins=J - 1)
#         plt.tick_params(labelbottom=False)
        plt.locator_params(axis='y', nbins=5)
        plt.title("Scattering spectrum \n (Modulus)", fontsize=fontsize)
        plt.legend(prop={'size': 15})

    def plot_phase(i_lb, y):
        for alpha in range(J - 1):
            betas = np.arange(-J + 1 + alpha, 0)
            plt.plot(betas, y[alpha, alpha:], label=fr'$a={alpha}$')
            plt.scatter(betas, y[alpha, alpha:], marker='+')
        plt.xticks(np.arange(-J + 1, 0), [(rf'${beta}$' if beta % 2 == 1 else '') for beta in np.arange(-J + 1, 0)],
                   fontsize=fontsize)
        plt.yticks(np.arange(-2, 3) * np.pi / 8,
                   [r'$-\frac{\pi}{4}$', r'$-\frac{\pi}{8}$', r'$0$', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$'],
                   fontsize=fontsize)
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xlabel(r'$b$', fontsize=fontsize)
#         if i_lb == 0:
#             plt.ylabel(r'$\mathrm{Phase}$', fontsize=fontsize)
#         else:
#             plt.tick_params(labelleft=False)
        plt.title("Scattering spectrum \n (Phase)", fontsize=fontsize)

    if axes is None:
        plt.figure(figsize=(max(len(labels), 5) * 3, 10))
    for i_lb, lb in enumerate(labels):
        if axes is not None:
            plt.sca(axes[0, i_lb])
            ax_mod = axes[0, i_lb]
        else:
            ax_mod = plt.subplot2grid((2, np.unique(i_graphs).size), (0, i_graphs[i_lb]))
        if i_lb == 0:
            ax_mod.yaxis.set_tick_params(which='major', direction='in', width=1.5, length=7)
            ax_mod.yaxis.set_label_coords(-0.18, 0.5)
        plot_modulus(i_lb, lb, C_env_mod[i_lb], model_bias[i_lb], estim_err[i_lb])

    for i_lb, lb in enumerate(labels):
        if axes is not None:
            plt.sca(axes[1, i_lb])
            ax_ph = axes[1, i_lb]
        else:
            ax_ph = plt.subplot2grid((2, np.unique(i_graphs).size), (1, i_graphs[i_lb]))
        plot_phase(i_lb, C_env_arg[i_lb])
        if i_lb == 0:
            ax_ph.yaxis.set_tick_params(which='major', direction='in', width=1.5, length=7)

    if axes is None:
        plt.tight_layout()

        leg = plt.legend(loc='upper center', ncol=1, fontsize=35, handlelength=1.0, labelspacing=1.0,
                         bbox_to_anchor=(1.3, 2.25, 0, 0))
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)


def plot_dashboard(RXs, estim_err=False, self_simi_err=False, bootstrap=True, theta_threshold=0.0075,
                   labels=None, linewidth=3.0, fontsize=20, ylim_phase=0.09, ylim_modulus=2.0, figsize=None):
    """
    Plot the scattering covariance dashboard for multi-scale processes composed of:
        - (wavelet power spectrum) sigma^2(j)
        - (sparsity factors) s^2(j)
        - (phase-envelope cross-spectrum) C_{W|W|}(a) as two graphs : |C_{W|W|}| and Arg(C_{W|W|})
        - (scattering cross-spectrum) C_S(a,b) as two graphs : |C_S| and Arg(C_S)

    :param RXs:
    :param estim_err:
    :param self_simi_err:
    :param bootstrap:
    :param theta_threshold:
    :param labels:
    :param linewidth:
    :param fontsize:
    :param ylim_phase:
    :param ylim_modulus:
    :param figsize:
    :return:
    """
    if isinstance(RXs, DescribedTensor):
        RXs = [RXs]
    fig, axes = plt.subplots(2, 2 + len(RXs), figsize=figsize or (10 + 5 * (len(RXs) - 1), 10))

    # marginal moments sigma^2 and s^2
    plot_marginal_moments(RXs, estim_err, axes[:, 0], labels, linewidth, fontsize)

    # phase-envelope cross-spectrum
    plot_cross_phased(RXs, estim_err, self_simi_err, theta_threshold, axes[:, 1], labels, fontsize, False, ylim_phase)

    # scattering cross spectrum
    plot_modulus(RXs, estim_err, self_simi_err, bootstrap, theta_threshold, axes[:, 2:], labels, fontsize, ylim_modulus)

    plt.tight_layout()
