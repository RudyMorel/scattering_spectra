""" Frontend functions for analysis and generation. """
import os
from pathlib import Path
from itertools import product
from time import time
import scipy
import numpy as np
import torch
import matplotlib.pyplot as plt

import utils.complex_utils as cplx
from utils import to_numpy
from frontend.data_source import ProcessDataLoader, FBmLoader, PoissonLoader, MRWLoader, SMRWLoader
from scattering_network.scale_indexer import ScaleIndexer
from scattering_network.time_layers import Wavelet, SpectrumNormalization
from scattering_network.moments import Marginal, Cov, CovStat
from scattering_network.module_chunk import ModuleChunk, SkipConnection
from scattering_network.described_tensor import DescribedTensor
from scattering_network.loss import MSELossScat
from scattering_network.solver import Solver, CheckConvCriterion, SmallEnoughException

""" Notations

Dimension sizes:
- B: number of batches (i.e. realizations of a process)
- N: number of data channels (N>1 : multivariate process)
- T: number of data samples (time samples)
- J: number of scales (octaves)
- Q: number of wavelets per octave
- r: number of conv layers in a scattering model

Tensor shapes:
- X: input, of shape  (B, N, T)
- RX: output (DescribedTensor), of shape (B, K, T, 2) with K the number of coefficients in the representation
"""


##################
# DATA LOADING
##################

def load_data(process_name, B, T, cache_dir=None, **data_param):
    """ Time series data loading function.

    :param process_name: fbm, poisson, mrw, smrw, hawkes, turbulence or snp
    :param B: number of realizations
    :param T: number of time samples
    :param cache_dir: the directory used to cache trajectories
    :param data_param: the model
    :return: dataloader
    """
    loader = {
        'fbm': FBmLoader,
        'poisson': PoissonLoader,
        'mrw': MRWLoader,
        'smrw': SMRWLoader,
    }

    if process_name == 'snp':
        raise ValueError("S&P data is private, please provide your own data.")
    if process_name == 'heliumjet':
        raise ValueError("Helium jet data is private, please provide your own data.")
    if process_name == 'hawkes':
        raise ValueError("Hawkes data is not yet supported.")
    if process_name not in loader.keys():
        raise ValueError("Unrecognized model name.")

    if cache_dir is None:
        cache_dir = Path(__file__).parents[0] / 'cached_dir'

    dtld = loader[process_name](cache_dir)
    X = dtld.load(B=B, T=T, **data_param).X

    return X[:, 0, :]


##################
# ANALYSIS
##################

def init_model(B, N, T, J, Q1, Q2, r_max, wav_type, high_freq, wav_norm,
               moments, m_types, qs, sigma,
               nchunks):
    """ Initialize a scattering covariance model.

    :param N: number of in_data channel
    :param T: number of time samples
    :param Q1: wavelets per octave first layer
    :param Q2: wavelets per octave second layer
    :param r_max: number convolution layers
    :param wav_type: wavelet type
    :param high_freq: central frequency of mother wavelet, 0.5 gives important aliasing
    :param wav_norm: wavelet normalization i.e. l1, l2
    :param moments: moments to compute on scattering, ex: None, 'marginal', 'cov', 'covstat'
    :param m_types: the type of moments to compute i.e. m00, m10, m11
    :param qs: if moments == 'marginal' the exponents of the marginal moments
    :param nchunks: the number of chunks

    :return: a torch module
    """
    module_list = []

    # scattering network
    sc_idxer = ScaleIndexer(J, Q1, Q2, r_max)
    W1 = Wavelet(T, J, Q1, wav_type, high_freq, wav_norm, 1, sc_idxer)
    module_list.append(W1)

    if moments == 'covstat' or sigma is not None:
        module_list.append(SpectrumNormalization(False, sigma))

    if r_max > 1:
        W2 = Wavelet(T, J, Q2, wav_type, high_freq, wav_norm, 2, sc_idxer)
        module_list.append(SkipConnection(W2))

    # moments
    if moments == 'marginal':
        module_list.append(Marginal(qs=qs or [1.0, 2.0]))
    if moments in ['cov', 'covstat']:
        module_list.append(Cov(N, sc_idxer, m_types))
    if moments == 'covstat':
        module_list.append(CovStat(J * Q1, m_types))

    model = ModuleChunk(module_list, B, N, nchunks)
    model.init_chunks()

    return model


def compute_sigma(X, B, T, J, Q1, Q2, wav_type, high_freq, wav_norm):
    """ Computes power specturm sigma(j)^2 used to normalize scattering coefficients. """
    marginal_model = init_model(B=B, N=1, T=T, J=J, Q1=Q1, Q2=Q2, r_max=1,
                                wav_type=wav_type, high_freq=high_freq, wav_norm=wav_norm,
                                moments='marginal', m_types=None, qs=[2.0], sigma=None,
                                nchunks=1)
    sigma = marginal_model(X).mean_batch()

    return sigma


def analyze(X, J=None, Q1=1, Q2=1, wav_type='battle_lemarie', wav_norm='l1', high_freq=0.425,
            moments='cov', m_types=None, nchunks=1, cuda=False):
    """ Compute sigma^2(j).
    :param X: an array of shape (T, ) or (B, T) or (B, N, T)
    :param J: number of octaves
    :param Q1: number of scales per octave on first wavelet layer
    :param Q2: number of scales per octave on second wavelet layer
    :param wav_type: wavelet type
    :param wav_norm: wavelet normalization i.e. l1, l2
    :param high_freq: central frequency of mother wavelet, 0.5 gives important aliasing
    :param moments: moments to compute on scattering, ex: None, 'marginal', 'cov', 'covstat'
    :param m_types: m00: sigma^2 and s^2, m10: cp, m11: cm
    :param nchunks: nb of chunks, increase it to reduce memory usage
    :param cuda: does calculation on gpu

    :return: a DescribedTensor result
    """
    if len(X.shape) == 1:  # assumes that X is of shape (T, )
        X = X[None, None, :]
    elif len(X.shape) == 2:  # assumes that X is of shape (B, T)
        X = X[:, None, :]

    B, N, T = X.shape
    X = cplx.from_np(X)

    if J is None:
        J = int(np.log2(T)) - 3

    # covstat needs a spectrum normalization
    sigma = compute_sigma(X, B, T, J, Q1, Q2, wav_type, high_freq, wav_norm) if moments == 'covstat' else None

    # initialize model
    model = init_model(B=B, N=N, T=T, J=J, Q1=Q1, Q2=Q2, r_max=2, wav_type=wav_type, high_freq=high_freq,
                       wav_norm=wav_norm, moments=moments,
                       m_types=m_types or ['m00', 'm10', 'm11'], qs=None, sigma=sigma,
                       nchunks=nchunks)

    # compute
    if cuda:
        X = X.cuda()
        model = model.cuda()

    RX = model(X)

    return RX.cpu()


##################
# GENERATION
##################

class GenDataLoader(ProcessDataLoader):
    """ A data loader for generation. Caches the generated trajectories. """
    def __init__(self, *args):
        super(GenDataLoader, self).__init__(*args)
        self.default_kwargs = {}

    def dirpath(self, **kwargs):
        """ The directory path in which the generated trajectories will be stored. """
        B_target = kwargs['X'].shape[0]
        model_params = kwargs['model_params']
        N, T, J, Q1, Q2, r_max, wav_type, m_types, moments = \
            (model_params[key] for key in ['N', 'T', 'J', 'Q1', 'Q2', 'r_max', 'wav_type', 'm_types', 'moments'])
        path_str = f"{self.model_name}_{wav_type}_B{B_target}_N{N}_T{T}_J{J}_Q1_{Q1}_Q2_{Q2}_rmax{r_max}_mo_{moments}" \
                   + f"{''.join(mtype[0] for mtype in m_types)}" \
                   + f"_tol{kwargs['optim_params']['tol_optim']:.2e}" \
                   + f"_it{kwargs['optim_params']['it']}"
        return self.dir_name / path_str.replace('.', '_').replace('-', '_')

    def generate_trajectory(self, X, RX, model_params, optim_params, gpu, dirpath):
        """ Performs cached generation. """
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        np.random.seed(None)
        filename = dirpath / str(np.random.randint(1e7, 1e8))
        if filename.is_file():
            raise OSError("File for saving this trajectory already exists.")

        X_torch = cplx.from_np(X)

        # sigma = None
        sigma = compute_sigma(X_torch, model_params['B'], model_params['T'], model_params['J'],
                              model_params['Q1'], model_params['Q2'],
                              model_params['wav_type'], model_params['high_freq'], model_params['wav_norm'])
        model_params['sigma'] = sigma

        # prepare target representation
        if RX is None:
            print("Preparing target representation")
            X_torch = cplx.from_np(X)
            model_avg = init_model(
                B=X.shape[0], nchunks=X.shape[0],
                **{key: value for (key, value) in model_params.items() if key not in ['B', 'nchunks']}
            )
            if gpu is not None:
                X_torch = X_torch.cuda()
                model_avg = model_avg.cuda()

            RX = model_avg(X_torch).mean_batch().cpu()

        # prepare initial gaussian process
        x0_mean = X.mean(-1).mean(0)
        x0_var = np.var(X, axis=-1).mean(0)

        def gen_wn(shape, mean, var):
            wn = np.random.randn(*shape)
            wn -= wn.mean(-1)
            wn /= np.std(wn, axis=-1)

            return wn * var + mean

        gen_shape = (1, ) + X.shape[1:]
        x0 = gen_wn(gen_shape, x0_mean, x0_var ** 0.5)

        # init model
        print("Initialize model")
        model = init_model(**model_params)

        # init loss, solver and convergence criterium
        loss = MSELossScat()
        solver_fn = Solver(model=model, loss=loss, xf=X[:, None, :, :], Rxf=RX, x0=x0.ravel(),
                           weights=None, cuda=optim_params['cuda'], relative_optim=optim_params['relative_optim'])

        check_conv_criterion = CheckConvCriterion(solver=solver_fn, tol=optim_params['tol_optim'])

        print('Embedding: uses {} coefficients {}'.format(
            model.count_coefficients(),
            ' '.join(
                ['{}={}'.format(m_type, model.count_coefficients(m_type=m_type))
                 for m_type in model.m_types])
        ))

        method, maxfun, jac = optim_params['method'], optim_params['maxfun'], optim_params['jac']
        relative_optim, it = optim_params['relative_optim'], optim_params['it']

        tic = time()
        # Decide if the function provides gradient or not
        func = solver_fn.joint if jac else solver_fn.function
        try:
            res = scipy.optimize.minimize(
                func, x0, method=method, jac=jac, callback=check_conv_criterion,
                options={'ftol': 1e-24, 'gtol': 1e-24, 'maxiter': it, 'maxfun': maxfun}
            )
            loss_tmp, x_opt, it, msg = res['fun'], res['x'], res['nit'], res['message']
        except SmallEnoughException:  # raised by check_conv_criterion
            print('SmallEnoughException')
            x_opt = check_conv_criterion.result
            it = check_conv_criterion.counter
            msg = "SmallEnoughException"

        toc = time()

        flo, fgr = solver_fn.joint(x_opt)
        flo, fgr = flo[0], np.max(np.abs(fgr))
        x_synt = x_opt.reshape(model_params['N'], -1)

        if not isinstance(msg, str):
            msg = msg.decode("ASCII")

        print('Optimization Exit Message : ' + msg)
        print(f"found parameters in {toc - tic:0.2f}s, {it} iterations -- {it / (toc - tic):0.2f}it/s")
        print(f"    abs sqrt error {flo ** 0.5:.2E}")
        print(f"    relative gradient error {fgr:.2E}")
        print(f"    loss0 {solver_fn.loss0:.2E}")

        return x_synt

    def worker(self, i, **kwargs):
        cuda = kwargs['optim_params']['cuda']
        gpus = kwargs['optim_params']['gpus']
        if cuda and gpus is None:
            kwargs['gpu'] = '0'
        elif gpus is not None:
            kwargs['gpu'] = str(gpus[i % len(gpus)])
        else:
            kwargs['gpu'] = None
        try:
            X = self.generate_trajectory(**kwargs)
            fname = f"{np.random.randint(1e7, 1e8)}.npy"
            np.save(str(kwargs['dirpath'] / fname), X[None, :, :])
        except ValueError as e:
            print(e)
            return


def generate(X, RX=None, S=1, J=None, Q1=1, Q2=1, wav_type='battle_lemarie', wav_norm='l1', high_freq=0.425,
             moments='cov', mtypes=None, qs=None, nchunks=1, it=10000, tol_optim=5e-4,
             generated_dir=None, exp_name=None, cuda=False, gpus=None, num_workers=1):
    """ Generate new realizations of X from a scattering covariance model.
    We first compute the scattering covariance representation of X and then sample it using gradient descent.

    :param X: an array of shape (T, ) or (B, T) or (B, N, T)
    :param RX: instead of X, the representation to generate from
    :param J: number of octaves
    :param Q1: number of scales per octave on first wavelet layer
    :param Q2: number of scales per octave on second wavelet layer
    :param wav_type: wavelet type
    :param wav_norm: wavelet normalization i.e. l1, l2
    :param high_freq: central frequency of mother wavelet, 0.5 gives important aliasing
    :param moments: moments to compute on scattering, ex: None, 'marginal', 'cov', 'covstat'
    :param qs: if moments == 'marginal' the exponents of the marginal moments
    :param nchunks: nb of chunks, increase it to reduce memory usage
    :param it: maximum number of gradient descent iteration
    :param tol_optim: error below which gradient descent stops
    :param generated_dir: the directory in which the generated dir will be located
    :param exp_name: experience name
    :param cuda: does calculation on gpu
    :param gpus: a list of gpus to use
    :param num_workers: number of generation workers

    :return: a DescribedTensor result
    """
    if len(X.shape) == 1:  # assumes that X is of shape (T, )
        X = X[None, None, :]
    elif len(X.shape) == 2:  # assumes that X is of shape (B, T)
        X = X[:, None, :]

    B, N, T = X.shape

    if J is None:
        J = int(np.log2(T)) - 5
    if mtypes is None:
        mtypes = ['m00', 'm10', 'm11']
    if qs is None:
        qs = [1.0, 2.0]
    if generated_dir is None:
        generated_dir = Path(__file__).parents[0] / 'cached_dir'

    # use a GenDataLoader to cache trajectories
    dtld = GenDataLoader(exp_name or 'gen_scat_cov', generated_dir, num_workers)

    # MODEL params
    model_params = {
        'B': 1, 'N': N, 'T': T, 'J': J, 'Q1': Q1, 'Q2': Q2, 'r_max': 2,
        'wav_type': wav_type,  # 'battle_lemarie' 'morlet' 'shannon'
        'high_freq': high_freq,  # 0.323645 or 0.425,
        'wav_norm': wav_norm,
        'moments': moments,
        'm_types': mtypes, 'qs': qs, 'sigma': None,
        'nchunks': nchunks,
    }

    # OPTIM params
    optim_params = {
        'it': it,
        'cuda': cuda or (gpus is not None),
        'gpus': gpus,
        'relative_optim': False,
        'maxfun': 2e6,
        'method': 'L-BFGS-B',
        'jac': True,  # origin of gradient, True: provided by solver, else estimated
        'tol_optim': tol_optim,
    }

    # multi-processed generation
    X_gen = dtld.load(B=S, X=X, RX=RX, model_params=model_params, optim_params=optim_params).X[:, 0, :]

    return X_gen


##################
# VIZUALIZE
##################

COLORS = ['skyblue', 'coral', 'lightgreen', 'darkgoldenrod', 'mediumpurple', 'red', 'purple', 'black',
          'paleturquoise'] + ['orchid'] * 20


def bootstrap_variance_complex(X, n_points, n_samples):
    """ Estimate variance of tensor X along last axis using bootstrap method. """
    # sample data uniformly
    sampling_idx = np.random.randint(low=0, high=X.shape[-2], size=(n_samples, n_points))
    sampled_data = X[..., sampling_idx, :]

    # computes mean
    mean = sampled_data.mean(-2).mean(-2)

    # computes bootstrap variance
    var = (cplx.modulus(sampled_data.mean(-2) - mean[..., None, :]).pow(2.0)).sum(-1) / (n_samples - 1)

    return mean, var


def error_arg(z_mod, z_err, eps=1e-12):
    """ Transform an error on |z| into an error on Arg(z). """
    z_mod = np.maximum(z_mod, eps)
    return np.arctan(z_err / z_mod / np.sqrt(np.clip(1 - z_err ** 2 / z_mod ** 2, 1e-6, 1)))


def get_variance(z):
    """ Compute complex variance of a sequence of complex numbers z1, z2, ... """
    B = z.shape[0]
    if z.shape[-1] != 2:
        z = cplx.from_real(z)
    return cplx.modulus(z - z.mean(0, keepdim=True)).pow(2.0).sum(0).div(B-1).div(B)


def plot_marginal_moments(RXs, estim_bar=False,
                          axes=None, labels=None, linewidth=3.0, fontsize=30):
    """ Plot the marginal moments
        - (wavelet power spectrum) sigma^2(j)
        - (sparsity factors) s^2(j)

    :param RXs: DescribedTensor or list of DescribedTensor
    :param estim_bar: display estimation error due to estimation on several realizations
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
    if axes is not None and axes.size != 2:
        raise ValueError("The axes provided to plot_marginal_moments should be an array of size 2.")

    labels = labels or [''] * len(RXs)
    axes = None if axes is None else axes.ravel()

    def get_data(RX, q):
        WX_nj = cplx.real(RX.select(r=1, m_type='m00', q=q, low=False)[:, :, 0])
        logWX_nj = torch.log2(WX_nj)
        return logWX_nj

    def plot_exponent(js, i_ax, ax, label, color, y, y_err):
        plt.sca(ax)
        plt.plot(-js, y, label=label, linewidth=linewidth, color=color)
        if not estim_bar:
            plt.scatter(-js, y, marker='+', s=200, linewidth=linewidth, color=color)
        else:
            eb = plt.errorbar(-js, y, yerr=y_err, capsize=4, color=color, fmt=' ')
            eb[-1][0].set_linestyle('--')
        plt.yscale('log', base=2)
        a, b = ax.get_ylim()
        if i_ax == 0:
            ax.set_ylim(min(a, 2**(-2)), max(b, 2**2))
        else:
            ax.set_ylim(min(2**(-2), a), 1.0)
        plt.xlabel(r'$-j$', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(-js, [fr'$-{j + 1}$' for j in js], fontsize=fontsize)

    if axes is None:
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        ax2 = plt.subplot2grid((1, 2), (0, 1))
        axes = [ax1, ax2]

    for i_lb, (lb, RX) in enumerate(zip(labels, RXs)):
        if 'm_type' not in RX.descri.columns:
            raise ValueError("The model output does not have the moments. ")
        columns = RX.descri.columns
        js = np.unique(RX.descri.reduce(r=1, low=False).j if 'j' in columns else RX.descri.reduce(low=False).j1)

        has_power_spectrum = 2.0 in RX.descri.q.values
        has_sparsity = 1.0 in RX.descri.q.values

        # averaging on the logs may have strange behaviors because of the strict convexity of the log
        # if you prefer to look at the log of the mean, then to a .mean_batch() on the representation before ploting it
        if has_power_spectrum:
            logWX2_n = get_data(RX, 2.0)
            logWX2_err = get_variance(logWX2_n) ** 0.5
            logWX2 = logWX2_n.mean(0)
            logWX2 -= logWX2[-1].item()
            plot_exponent(js, 0, axes[0], lb, COLORS[i_lb], 2.0 ** logWX2, np.log(2) * logWX2_err * 2.0 ** logWX2)
            if i_lb == len(labels) - 1 and any([lb != '' for lb in labels]):
                plt.legend(prop={'size': 15})
            plt.title('Wavelet Spectrum', fontsize=fontsize)

            if has_sparsity:
                logWX1_n = get_data(RX, 1.0)
                logWXs_n = 2 * logWX1_n - logWX2_n.mean(0, keepdims=True)
                logWXs_err = get_variance(logWXs_n) ** 0.5
                logWXs = logWXs_n.mean(0)
                plot_exponent(js, 1, axes[1], lb, COLORS[i_lb], 2.0 ** logWXs, np.log(2) * logWXs_err * 2.0 ** logWXs)
                if i_lb == len(labels) - 1 and any([lb != '' for lb in labels]):
                    plt.legend(prop={'size': 15})
                plt.title('Sparsity factor', fontsize=fontsize)


def plot_phase_envelope_spectrum(RXs, estim_bar=False, self_simi_bar=False, theta_threshold=0.005,
                                 axes=None, labels=None, fontsize=30, single_plot=False, ylim=0.09):
    """ Plot the phase-envelope cross-spectrum C_{W|W|}(a) as two graphs : |C_{W|W|}| and Arg(C_{W|W|}).

    :param RXs: DescribedTensor or list of DescribedTensor
    :param estim_bar: display estimation error due to estimation on several realizations
    :param self_simi_bar: display self-similarity error, it is a measure of scale regularity
    :param theta_threshold: rules phase instability
    :param axes: custom axes: array of size 2
    :param labels: list of labels for each model output
    :param fontsize: labels fontsize
    :param single_plot: output all DescribedTensor on a single plot
    :param ylim: above y limit of modulus graph
    :return:
    """
    if isinstance(RXs, DescribedTensor):
        RXs = [RXs]
    if labels is not None and len(RXs) != len(labels):
        raise ValueError("Invalid number of labels")

    labels = labels or [''] * len(RXs)
    columns = RXs[0].descri.columns
    J = RXs[0].descri.j.max() if 'j' in columns else RXs[0].descri.j1.max()

    c_wmw = torch.zeros(len(labels), J-1, 2)
    err_estim = torch.zeros(len(labels), J-1)
    err_self_simi = torch.zeros(len(labels), J-1)

    for i_lb, RX in enumerate(RXs):
        # infer model type
        if ('j1' in RX.descri) and ('jp1' in RX.descri):
            model_type = 'cov'
        elif 'j' in RX.descri:
            model_type = 'covstat'
        else:
            continue

        B = RX.y.shape[0]

        sigma = cplx.real(RX.select(r=1, q=2, low=False)[:, :, 0, :]).unsqueeze(-1).mean(0, keepdim=True)

        for a in range(1, J):
            if model_type == 'covstat':
                c_mwm_n = RX.select(m_type='m10', a=a, low=False)[:, 0, 0, :]
                c_wmw[i_lb, a-1, :] = c_mwm_n.mean(0)
                err_estim[i_lb, a-1] = get_variance(c_mwm_n).pow(0.5)
            else:
                c_mwm_nj = torch.zeros(B, J-a, 2)
                for j1 in range(a, J):
                    coeff = RX.select(m_type='m10', j1=j1-a, jp1=j1, low=False)[:, 0, 0, :]
                    coeff /= sigma[:, j1, ...].pow(0.5) * sigma[:, j1-a, ...].pow(0.5)
                    c_mwm_nj[:, j1 - a, :] = coeff

                # the mean in j of the variance of time estimators
                c_wmw[i_lb, a-1, :] = c_mwm_nj.mean(0).mean(0)
                err_self_simi_n = (cplx.modulus(c_mwm_nj).pow(2.0).mean(1) - cplx.modulus(c_mwm_nj.mean(1)).pow(2.0)) / c_mwm_nj.shape[1]
                err_self_simi[i_lb, a-1] = err_self_simi_n.mean(0).pow(0.5)
                err_estim[i_lb, a-1] = get_variance(c_mwm_nj.mean(1)).pow(0.5)

    c_wmw_mod, cwmw_arg = np.abs(cplx.to_np(c_wmw)), np.angle(cplx.to_np(c_wmw))
    err_self_simi, err_estim = to_numpy(err_self_simi), to_numpy(err_estim)
    err_self_simi_arg, err_estim_arg = error_arg(c_wmw_mod, err_self_simi), error_arg(c_wmw_mod, err_estim)

    # phase instability at z=0
    for z_arg in [cwmw_arg, err_self_simi_arg, err_estim_arg]:
        z_arg[c_wmw_mod < theta_threshold] = 0.0

    def plot_modulus(i_lb, label, color, y, y_err_estim, y_err_self_simi):
        a_s = np.arange(1, J)
        plt.plot(a_s, y, color=color or 'green', label=label)
        if not estim_bar and not self_simi_bar:
            plt.scatter(a_s, y, color=color or 'green', marker='+')
        if self_simi_bar:
            plot_x_offset = -0.07 if estim_bar else 0.0
            plt.errorbar(a_s + plot_x_offset, y, yerr=y_err_self_simi, capsize=4, color=color, fmt=' ')
        if estim_bar:
            plot_x_offset = 0.07 if self_simi_bar else 0.0
            eb = plt.errorbar(a_s + plot_x_offset, y, yerr=y_err_estim, capsize=4, color=color, fmt=' ')
            eb[-1][0].set_linestyle('--')
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

    def plot_phase(label, color, y, y_err_estim, y_err_self_simi):
        a_s = np.arange(1, J)
        plt.plot(a_s, y, color=color, label=label)
        if not estim_bar and not self_simi_bar:
            plt.scatter(a_s, y, color=color, marker='+')
        if self_simi_bar:
            plot_x_offset = -0.07 if estim_bar else 0.0
            plt.errorbar(a_s + plot_x_offset, y, yerr=y_err_self_simi, capsize=4, color=color, fmt=' ')
        if estim_bar:
            plot_x_offset = 0.07 if self_simi_bar else 0.0
            eb = plt.errorbar(a_s + plot_x_offset, y, yerr=y_err_estim, capsize=4, color=color, fmt=' ')
            eb[-1][0].set_linestyle('--')
        plt.xticks(np.arange(1, J), [(rf'${j}$' if j % 2 == 1 else '') for j in np.arange(1, J)], fontsize=fontsize)
        plt.yticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi],
                   [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=fontsize)
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xlabel(r'$a$', fontsize=fontsize)
        plt.title("Phase-Env spectrum \n (Phase)", fontsize=fontsize)

    if axes is None:
        plt.figure(figsize=(5, 10) if single_plot else (len(labels) * 5, 10))
        ax_mod = plt.subplot2grid((2, 1), (0, 0))
        ax_mod.yaxis.set_tick_params(which='major', direction='in', width=1.5, length=7)
    else:
        plt.sca(axes[0])
    for i_lb, lb in enumerate(labels):
        plot_modulus(i_lb, lb, COLORS[i_lb], c_wmw_mod[i_lb], err_estim[i_lb], err_self_simi[i_lb])
        if i_lb == len(labels) - 1 and any([lb != '' for lb in labels]):
            plt.legend(prop={'size': 15})

    if axes is None:
        plt.subplot2grid((2, 1), (1, 0))
    else:
        plt.sca(axes[1])
    for i_lb, lb in enumerate(labels):
        plot_phase(lb, COLORS[i_lb], cwmw_arg[i_lb], err_estim_arg[i_lb], err_self_simi_arg[i_lb])

    if axes is None:
        plt.tight_layout()


def plot_scattering_spectrum(RXs, estim_bar=False, self_simi_bar=False, bootstrap=True, theta_threshold=0.01,
                             axes=None, labels=None, fontsize=40, ylim=2.0):
    """ Plot the scattering cross-spectrum C_S(a,b) as two graphs : |C_S| and Arg(C_S).

    :param RXs: DescribedTensor or list of DescribedTensor
    :param estim_bar: display estimation error due to estimation on several realizations
    :param self_simi_bar: display self-similarity error, it is a measure of scale regularity
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
    if axes is not None and axes.size != 2 * len(RXs):
        raise ValueError(f"Existing axes must be provided as an array of size {2 * len(RXs)}")

    axes = None if axes is None else axes.reshape(2, len(RXs))

    labels = labels or [''] * len(RXs)
    i_graphs = np.arange(len(labels))

    columns = RXs[0].descri.columns
    J = RXs[0].descri.j.max() if 'j' in columns else RXs[0].descri.j1.max()

    cs = torch.zeros(len(labels), J-1, J-1, 2)
    err_estim = torch.zeros(len(labels), J - 1, J - 1)
    err_self_simi = torch.zeros(len(labels), J-1, J-1)

    for i_lb, (RX, lb, color) in enumerate(zip(RXs, labels, COLORS)):
        # infer model type
        if ('j1' in RX.descri) and ('jp1' in RX.descri):
            model_type = 'cov'
        elif 'j' in RX.descri:
            model_type = 'covstat'
        else:
            continue

        if self_simi_bar and model_type == 'covstat':
            raise ValueError("Impossible to output self-similarity error on covstat model. Use a cov model instead.")

        B = RX.y.shape[0]

        sigma = cplx.real(RX.select(r=1, q=2, low=False)[:, :, 0, :]).unsqueeze(-1).mean(0, keepdim=True)

        for (a, b) in product(range(J - 1), range(-J + 1, 0)):
            if a - b >= J:
                continue

            # prepare covariances
            if model_type == 'cov':
                cs_nj = torch.zeros(B, J+b-a, 2)
                for j1 in range(a, J+b):
                    coeff = RX.select(m_type='m11', j1=j1, jp1=j1-a, j2=j1-b, low=False)[:, 0, 0, :]
                    coeff /= sigma[:, j1, ...].pow(0.5) * sigma[:, j1 - a, ...].pow(0.5)
                    cs_nj[:, j1 - a, :] = coeff

                cs_j = cs_nj.mean(0)
                cs[i_lb, a, J - 1 + b, :] = cs_j.mean(0)
                if b == -J + a + 1:
                    err_self_simi[i_lb, a, J - 1 + b] = 0.0
                else:
                    err_self_simi[i_lb, a, J - 1 + b] = cplx.modulus(cs_j - cs_j.mean(0, keepdim=True)) \
                        .pow(2.0).sum(0).div(J + b - a - 1).pow(0.5)
                # compute estimation error
                if bootstrap:
                    # mean, var = bootstrap_variance_complex(cs_nj.transpose(0, 1), cs_nj.shape[0], 20000)
                    mean, var = bootstrap_variance_complex(cs_nj.mean(1), cs_nj.shape[0], 20000)
                    err_estim[i_lb, a, J - 1 + b] = var.pow(0.5)
                else:
                    err_estim[i_lb, a, J - 1 + b] = (cplx.modulus(cs_nj).pow(2.0).mean(0) -
                                                     cplx.modulus(cs_nj.mean(0)).pow(2.0)) / (B - 1)
            else:
                coeff_ab = RX.select(m_type='m11', a=a, b=b, low=False)[:, 0, 0, :]
                cs[i_lb, a, J-1+b, :] = coeff_ab.mean(0)

    cs, cs_mod, cs_arg = cplx.to_np(cs), np.abs(cplx.to_np(cs)), np.angle(cplx.to_np(cs))
    err_self_simi, err_estim = to_numpy(err_self_simi), to_numpy(err_estim)
    err_self_simi_arg, err_estim_arg = error_arg(cs_mod, err_self_simi), error_arg(cs_mod, err_estim)

    # power spectrum normalization
    bs = np.arange(-J + 1, 0)[None, :]
    cs_mod /= (2.0 ** bs)
    err_self_simi /= (2.0 ** bs)
    err_estim /= (2.0 ** bs)

    # phase instability at z=0
    for z_arg in [cs_arg, err_self_simi_arg, err_estim_arg]:
        z_arg[cs_mod < theta_threshold] = 0.0

    def plot_modulus(label, y, y_err_estim, y_err_self_simi):
        for a in range(J-1):
            bs = np.arange(-J+1+a, 0)
            line = plt.plot(bs, y[a, a:], label=label if a == 0 else '')
            color = line[-1].get_color()
            if not estim_bar and not self_simi_bar:
                plt.scatter(bs, y[a, a:], marker='+')
            if self_simi_bar:
                plot_x_offset = -0.07 if self_simi_bar else 0.0
                plt.errorbar(bs + plot_x_offset, y[a, a:],
                             yerr=y_err_self_simi[a, a:], capsize=4, color=color, fmt=' ')
            if estim_bar:
                plot_x_offset = 0.07 if self_simi_bar else 0.0
                eb = plt.errorbar(bs + plot_x_offset, y[a, a:],
                                  yerr=y_err_estim[a, a:], capsize=4, color=color, fmt=' ')
                eb[-1][0].set_linestyle('--')
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xticks(np.arange(-J + 1, 0), [(rf'${b}$' if b % 2 == 1 else '') for b in np.arange(-J+1, 0)],
                   fontsize=fontsize)
        plt.xlabel(r'$b$', fontsize=fontsize)
        plt.ylim(-0.02, ylim)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.yticks(fontsize=fontsize)
        plt.locator_params(axis='x', nbins=J - 1)
        plt.locator_params(axis='y', nbins=5)
        plt.title("Scattering spectrum \n (Modulus)", fontsize=fontsize)
        if label != '':
            plt.legend(prop={'size': 15})

    def plot_phase(y, y_err_estim, y_err_self_simi):
        for a in range(J-1):
            bs = np.arange(-J+1+a, 0)
            line = plt.plot(bs, y[a, a:], label=fr'$a={a}$')
            color = line[-1].get_color()
            if not estim_bar and not self_simi_bar:
                plt.scatter(bs, y[a, a:], marker='+')
            if self_simi_bar:
                plot_x_offset = -0.07 if estim_bar else 0.0
                plt.errorbar(bs + plot_x_offset, y[a, a:],
                             yerr=y_err_self_simi[a, a:], capsize=4, color=color, fmt=' ')
            if estim_bar:
                plot_x_offset = 0.07 if self_simi_bar else 0.0
                eb = plt.errorbar(bs + plot_x_offset, y[a, a:],
                                  yerr=y_err_estim[a, a:], capsize=4, color=color, fmt=' ')
                eb[-1][0].set_linestyle('--')
        plt.xticks(np.arange(-J+1, 0), [(rf'${b}$' if b % 2 == 1 else '') for b in np.arange(-J+1, 0)],
                   fontsize=fontsize)
        plt.yticks(np.arange(-2, 3) * np.pi / 8,
                   [r'$-\frac{\pi}{4}$', r'$-\frac{\pi}{8}$', r'$0$', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$'],
                   fontsize=fontsize)
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xlabel(r'$b$', fontsize=fontsize)
        plt.title("Scattering spectrum \n (Phase)", fontsize=fontsize)

    if axes is None:
        plt.figure(figsize=(max(len(labels), 5) * 3, 10))
    for i_lb, lb in enumerate(labels):
        if axes is not None:
            plt.sca(axes[0, i_lb])
            ax_mod = axes[0, i_lb]
        else:
            ax_mod = plt.subplot2grid((2, np.unique(i_graphs).size), (0, i_graphs[i_lb]))
        ax_mod.yaxis.set_tick_params(which='major', direction='in', width=1.5, length=7)
        ax_mod.yaxis.set_label_coords(-0.18, 0.5)
        plot_modulus(lb, cs_mod[i_lb], err_estim[i_lb], err_self_simi[i_lb])

    for i_lb, lb in enumerate(labels):
        if axes is not None:
            plt.sca(axes[1, i_lb])
            ax_ph = axes[1, i_lb]
        else:
            ax_ph = plt.subplot2grid((2, np.unique(i_graphs).size), (1, i_graphs[i_lb]))
        plot_phase(cs_arg[i_lb], err_estim_arg[i_lb], err_self_simi_arg[i_lb])
        if i_lb == 0:
            ax_ph.yaxis.set_tick_params(which='major', direction='in', width=1.5, length=7)

    if axes is None:
        plt.tight_layout()
        leg = plt.legend(loc='upper center', ncol=1, fontsize=35, handlelength=1.0, labelspacing=1.0,
                         bbox_to_anchor=(1.3, 2.25, 0, 0))
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)


def plot_dashboard(RXs, estim_bar=False, self_simi_bar=False, bootstrap=True, theta_threshold=None,
                   labels=None, linewidth=3.0, fontsize=20, ylim_phase=0.09, ylim_modulus=2.0, figsize=None, axes=None):
    """ Plot the scattering covariance dashboard for multi-scale processes composed of:
        - (wavelet power spectrum) sigma^2(j)
        - (sparsity factors) s^2(j)
        - (phase-envelope cross-spectrum) C_{W|W|}(a) as two graphs : |C_{W|W|}| and Arg(C_{W|W|})
        - (scattering cross-spectrum) C_S(a,b) as two graphs : |C_S| and Arg(C_S)

    :param RXs: DescribedTensor or list of DescribedTensor
    :param estim_bar: display estimation error due to estimation on several realizations
    :param self_simi_bar: display self-similarity error, it is a measure of scale regularity
    :param bootstrap: time variance computation method
    :param theta_threshold: rules phase instability
    :param labels: list of labels for each model output
    :param linewidth: lines linewidth
    :param fontsize: labels fontsize
    :param ylim_phase: graph ylim for the phase
    :param ylim_modulus: graph ylim for the modulus
    :param figsize: figure size
    :param axes: custom array of axes, should be of shape (2, 2 + nb of representation to plot)
    :return:
    """
    if theta_threshold is None:
        theta_threshold = [0.005, 0.1]
    if isinstance(RXs, DescribedTensor):
        RXs = [RXs]
    for RX in RXs:
        if 'n1p' in RX.descri.columns:
            ns = RX.descri.to_array(['n1', 'n1p'])
        else:
            ns = RX.descri.to_array(['n1'])
        ns_unique = set([tuple(n) for n in ns])
        if len(ns_unique) != 1:
            raise ValueError("Plotting functions do not support multi-variate representation other than "
                             "univariate or single pair.")

    if axes is None:
        _, axes = plt.subplots(2, 2 + len(RXs), figsize=figsize or (10 + 5 * (len(RXs) - 1), 10))

    # marginal moments sigma^2 and s^2
    plot_marginal_moments(RXs, estim_bar, axes[:, 0], labels, linewidth, fontsize)

    # phase-envelope cross-spectrum
    plot_phase_envelope_spectrum(RXs, estim_bar, self_simi_bar, theta_threshold[0], axes[:, 1], labels, fontsize, False, ylim_phase)

    # scattering cross spectrum
    plot_scattering_spectrum(RXs, estim_bar, self_simi_bar, bootstrap, theta_threshold[1], axes[:, 2:], labels, fontsize, ylim_modulus)

    plt.tight_layout()
