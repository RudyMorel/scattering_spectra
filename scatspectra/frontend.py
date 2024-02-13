""" Frontend functions for analysis and generation.

Notations.

Dimension sizes:
- B: data batch size
- R: number of realizations (used to estimate scattering covariance)
- N: number of data channels (N>1 : multivariate process)
- T: number of data samples (time samples)
- J: number of scales (octaves)
- Q: number of wavelets per octave (Q>1 means better sampling of the scales)
- r: number of convolutional layers in a scattering model

Tensor shapes:
- x: input, of shape  (B, N, T)
- Rx: output (DescribedTensor), 
    - y: tensor of shape (B, K, T) with K the number of coefficients in the representation
    - descri: pandas DataFrame of shape K x nb_attributes used, description of the output tensor y
"""
import os
from itertools import product
from time import time
import numpy as np
import scipy
import pandas as pd
import torch
import matplotlib.pyplot as plt

from scatspectra.utils import to_numpy
from scatspectra.data_source import (
    DataGeneratorBase,
    PoissonGenerator, FBmGenerator, MRWGenerator, SMRWGenerator, SPDaily
)
from scatspectra.models import Model, ADMISSIBLE_MODEL_TYPES
from scatspectra.layers import (
    DescribedTensor, MSELossScat, format_np, Solver, CheckConvCriterion, 
    SmallEnoughException, MaxIteration,
)
from scatspectra.description import make_description_compatible


##################
# DATA LOADING
##################

def load_data(name, R=None, T=None,
              cache_path=None, num_workers=1,
              **model_params):
    """ Load log-prices from standard models e.g. fBm, Poisson, MRW, SMRW.

    :param model_name: fbm, poisson, mrw, smrw, hawkes, turbulence or snp
    :param R: number of realizations
    :param T: number of time samples
    :param cache_path: the directory used to cache data
    :param num_workers: set num_workers > 1 for parallel generation
    :param model_params: model parameters (e.g. intermittency parameter for MRW)
    :return: dataloader
    """
    if name == 'snp':
        return SPDaily().x

    generator = {
        'fbm': FBmGenerator,
        'poisson': PoissonGenerator,
        'mrw': MRWGenerator,
        'smrw': SMRWGenerator,
    }

    if name == 'heliumjet':
        raise ValueError("Helium jet data is private.")
    if name == 'hawkes':
        raise ValueError("Hawkes data is not yet supported.")
    if name not in generator.keys():
        raise ValueError("Unrecognized model name.")
    
    data_gen = generator[name](
        cache_path=cache_path, T=T, **model_params
    )
    x = data_gen.load(R, num_workers)

    return x


##################
# ANALYSIS
##################


def compute_sigma2(x, J, Q, wav_type, wav_norm, high_freq, rpad, cuda, nchunks):
    """Computes sigma(j)^2 = <|Wx(t,j)|^2>_t used to normalize wavelet coefficients. 

    :param x: input tensor of shape (batch_size, in_channels, ..., T)
    :param J: 
    :param Q:
    :param wav_type: _description_
    :param wav_norm: _description_
    :param high_freq: _description_
    :param rpad: _description_
    :param cuda: _description_
    :return: _description_
    """
    # initialize model
    model = Model(model_type="scat_marginal", T=x.shape[-1], r=1, J=J, Q=Q,
                  wav_type=wav_type, wav_norm=wav_norm, high_freq=high_freq,
                  A=None, rpad=rpad, channel_transforms=None, N=x.shape[1],
                  Ns=None, diago_n=True, cross_params=None,
                  sigma2=None, sigma2_L1=None, sigma2_L2=None,
                  sigma2_Lphix=None, norm_on_the_fly=False,
                  estim_operator=None, qs=[2.0], coeff_types=None,
                  dtype=x.dtype, histogram_moments=False,
                  skew_redundance=False, nchunks=nchunks)

    if cuda:
        x = x.cuda()
        model = model.cuda()

    # compute sigma2
    sigma2 = model(x).y.real.reshape(x.shape[0], x.shape[1], -1)  # (B, N, J)

    return sigma2


def analyze(x, model_type="scat_spectra", r=2,
            J=None, Q=1, wav_type="battle_lemarie", wav_norm="l1",
            high_freq=0.425, rpad=True,
            normalize='batch_ps', sigma2=None, keep_ps=True,
            diago_n=True, cross_params=None,
            qs=[1.0, 2.0], estim_operator=None,
            histogram_moments=False, skew_redundance=True,
            cuda=False, nchunks=1):
    """ Compute scattering based model.

    :param x: an array of shape (T, ), (B, T) or (B, N, T)
    :param model_type: moments to compute on scattering
        None: compute Sx = (Wx, W|Wx|) and keep time axis t
        "scat_marginal": compute marginal statistics on Sx: <|Sx|^q>_t
        "scat_spectra": compute scattering spectra which corresponds to 
            <Sx>_t and <Sx Sx^T>_t
        "inv_scat_spectra": invariant scattering spectra, same as "scat_spectra" but now 
            consider the self-similar projection of matrix <Sx Sx^T>_t
        "scat_marginal+scat_spectra": both "scat_marginal" and "scat_spectra"
    :param r: number of wavelet layers
    :param J: number of octaves for each wavelet layer
    :param Q: number of scales per octave for each wavelet layer
    :param wav_type: wavelet types for each wavelet layer
    :param wav_norm: wavelet normalization i.e. l1, l2 for each layer
    :param high_freq: central frequency of mother wavelet for each layer, 0.5 may lead to important aliasing
    :param rpad: if true, uses a reflection pad to account for edge effects
    :param normalize:
        None: no normalization,
        'each_ps': normalize wavelet coefficients by their energy on each time-series 
        'batch_ps': normalize wavelet coefficients by their energy averaged on the batch
    :param sigma2: a tensor of size B x N x J, override normalization
    :param keep_ps: keep the power spectrum even after normalization
    :paran diago_n: diagonal model along input channels
    :param cross_params: dictionary containing cross-cov model parameters
    :param qs: exponent to use in a "scat_marginal" or "scat+scat_spectra" model
    :param estim_operator: the operator computing the average on time <.>_t by,
        uniform average by default. 
    :param cuda: does calculation on gpu
    """
    if model_type not in ADMISSIBLE_MODEL_TYPES:
        raise ValueError(f"Unrecognized model type {model_type}.")
    if normalize not in [None, 'each_ps', 'batch_ps']:
        raise ValueError("Unrecognized normalization.")
    if model_type == "inv_scat_spectra" and normalize is None:
        raise ValueError("For inv_scat_spectra model, user should provide a "
                         "normalize argument.")
    if model_type == "cross-cov" and normalize not in ['each_ps', 'batch_ps']:
        raise ValueError("Cross-cov model should be normalized.")
    if r > 2 and model_type not in [None, "scat_marginal"]:
        raise ValueError("Scattering Spectra are not implemented" +
                         "for more than 3 convolution layers.")

    # format input
    x = torch.tensor(format_np(x))
    B, N, T = x.shape

    if x.shape[1] > 1 and not diago_n:
        raise ValueError("Multi-variate is currently under development.")

    if x.dtype not in [torch.float32, torch.float64]:
        x = x.type(torch.float32)
        print("WARNING. Casting data to torch.float32.")

    # default value
    if J is None:
        J = int(np.log2(T)) - 3

    # compute normalization
    if normalize is not None and sigma2 is None:
        sigma2 = compute_sigma2(
            x, J, Q, wav_type, wav_norm, high_freq, rpad, cuda, nchunks
        )
        if normalize == 'batch_ps':
            sigma2 = sigma2.mean(0, keepdim=True)
    if sigma2 is not None and sigma2.is_complex():
        raise ValueError("Normalization should be real!.")

    # initialize model
    model = Model(model_type=model_type, T=T, r=r, J=J, Q=Q,
                  wav_type=wav_type, wav_norm=wav_norm, high_freq=high_freq,
                  A=None, rpad=rpad, channel_transforms=None, N=N,
                  Ns=None, diago_n=diago_n, cross_params=cross_params,
                  sigma2=sigma2, norm_on_the_fly=False,
                  estim_operator=estim_operator, qs=qs, coeff_types=None,
                  dtype=x.dtype, histogram_moments=histogram_moments,
                  skew_redundance=skew_redundance, nchunks=nchunks)

    # compute
    if cuda:
        x = x.cuda()
        model = model.cuda()
    Rx = model(x)
    Rx.config = model.config

    # add back any information that may have been normalized
    if keep_ps and sigma2 is not None \
            and model_type in ["scat_spectra", "scat+scat_spectra", "inv_scat_spectra"]:
        # 1. the mean: <x>_t
        #    otherwise would output <x>_t / <|Wx(t,J)|^2>^0.5
        mask_mean = Rx.df.eval(f"coeff_type=='mean'").values
        if mask_mean.sum() != 0:
            sigma2_bJ = sigma2[:, :, -1].reshape(sigma2.shape[0], -1, 1)
            Rx.y[:, mask_mean, :] = Rx.y[:, mask_mean, :] * sigma2_bJ.pow(0.5)
        # 2. the variances: <|Wx(t,j)|^2>_t
        #    otherwise would output 1.0 = <|Wx(t,j)|^2>_t / <|Wx(t,j)|^2>_t
        for n in range(N):
            mask_ps = (
                Rx.df
                .eval(f"coeff_type=='variance' & nl=={n} & nr=={n}")
                .values
            )
            if mask_ps.sum() != 0:
                sigma2_bj = sigma2[:, n, :].reshape(sigma2.shape[0], -1, 1)
                Rx.y[:, mask_ps, :] = Rx.y[:, mask_ps, :] * sigma2_bj

    return Rx.cpu()


def format_to_real(Rx):
    """ Transforms a complex described tensor z into a real tensor (Re z, Im z). """
    if "is_real" not in Rx.df:
        raise ValueError("Described tensor should have a column indicating " +
                         "which coefficients are real.")
    Rx_real = Rx.query(is_real=True)
    Rx_complex = Rx.query(is_real=False)

    # new description
    df_complex_real = Rx_complex.df.copy()
    df_complex_imag = Rx_complex.df.copy()
    df_complex_real["is_real"] = True
    df = pd.concat([Rx_real.df, df_complex_real, df_complex_imag])

    # new tensor
    y = torch.cat([
        Rx_real.y.real, Rx_complex.y.real, Rx_complex.y.imag
    ], dim=1)

    return DescribedTensor(x=Rx.x, y=y, df=df)


def self_simi_obstruction_score(x, Rx=None, J=None, Q=1,
                                wav_type='battle_lemarie', wav_norm='l1', high_freq=0.425,
                                nchunks=1, cuda=False):
    """ Quantifies obstruction to self-similarity in a certain range of scales.

    :param x: an array of shape (T, ) or (B, T) or (B, N, T)
    :param Rx: overwrite representation on which to assess self-similarity, should be a normalized representation
    :param J: number of octaves for each wavelet layer
    :param Q: number of scales per octave for each wavelet layer
    :param wav_type: wavelet types for each wavelet layer
    :param wav_norm: wavelet normalization i.e. l1, l2 for each layer
    :param high_freq: central frequency of mother wavelet for each layer, 0.5 gives important aliasing
    :param nchunks: nb of chunks, increase it to reduce memory usage
    :param cuda: does calculation on gpu

    :return:
        - score on white noise reference (gives the score estimation error)
        - score on x
    """
    if Rx is None:
        Rx = analyze(x, model_type='scat_spectra', r=2, J=J, Q=Q,
                     wav_type=wav_type, wav_norm=wav_norm, high_freq=high_freq,
                     normalize='batch_ps',
                     estim_operator=None, cuda=cuda, nchunks=nchunks).mean_batch()

    # white noise reference score
    Rx_wn = None
    if x is not None:
        x_wn = np.random.randn(*x.shape)
        Rx_wn = analyze(x_wn, model_type='scat_spectra', r=2, J=J, Q=Q,
                        wav_type=wav_type, wav_norm=wav_norm, high_freq=high_freq,
                        normalize='batch_ps',
                        estim_operator=None, cuda=cuda, nchunks=nchunks).mean_batch()

    def self_simi_score_spars(Rx):
        Wx1 = Rx.query(coeff_type='spars', is_low=False).y[:, :, 0]
        logWxs = Wx1.real.pow(2.0).log2()
        dlogWxs = logWxs[:, 1:] - logWxs[:, :-1]
        return 1e1 * dlogWxs.std(-1).numpy()

    def self_simi_score_ps(Rx):
        Wx2 = Rx.query(coeff_type='variance', is_low=False).y[:, :, 0]
        logWx2 = Wx2.real.log2()
        dlogWx2 = logWx2[:, 1:] - logWx2[:, :-1]
        return 2e1 * dlogWx2.std(-1).numpy()

    def self_simi_score_phase_mod(Rx):
        J = Rx.df.j.max() if 'j' in Rx.df.columns else Rx.df.jl1.max()

        score = np.zeros(Rx.y.shape[0])
        for a in range(1, J - 1):
            phi3 = torch.stack([Rx.query(coeff_type='skewness', jl1=j1, jr1=j1 - a, is_low=False).y[0, 0, 0]
                                for j1 in range(a, J)])
            score += phi3.numpy().std(-1)

        return 1e1 * score / (J - 1)

    def self_simi_score_mod(Rx):
        J = Rx.df.j.max() if 'j' in Rx.df.columns else Rx.df.jl1.max()

        ndiagonals = 0
        score = np.zeros(Rx.y.shape[0])
        for (a, b) in product(range(J - 1), range(-J + 1, 0)):
            if a - b >= J - 1:
                continue
            phi4 = torch.stack([Rx.query(coeff_type='kurtosis', jl1=j1, jr1=j1 - a, j2=j1 - b, is_low=False).y[0, 0, 0]
                                for j1 in range(a, J + b)])
            ndiagonals += 1
            score += phi4.numpy().std(-1)

        return 1e2 * score / ndiagonals

    def get_score(Rx):
        score_phi1 = self_simi_score_spars(Rx)
        score_phi2 = self_simi_score_ps(Rx)
        score_phi3 = self_simi_score_phase_mod(Rx)
        score_phi4 = self_simi_score_mod(Rx)
        return {
            'spars': score_phi1,
            'variance': score_phi2,
            'skewness': score_phi3,
            'kurtosis': score_phi4,
            'total': score_phi1 + score_phi2 + score_phi3 + score_phi4
        }

    return None if x is None else get_score(Rx_wn), get_score(Rx)


##################
# GENERATION
##################

class ScatGenerator(DataGeneratorBase):
    """ A data loader for generation. Caches the generated time-series. """

    def __init__(self, B, cache_path, **kwargs):
        super(ScatGenerator, self).__init__(model_name="scattering",
                                            B=B,
                                            cache_path=cache_path,
                                            **kwargs)

    def generate_batch(self, i) -> np.ndarray:
        """ Generate a batch of data. """

        config = self.config

        # PREPARATION
        # set gpu
        gpus = config['gpus']
        if gpus is not None:
            gpu = gpus[i % len(gpus)]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # set seed
        np.random.seed(config['seed'])

        x_observed = None
        if config['x'] is not None:
            x_observed = torch.tensor(config['x'])

        # shape to generate
        B, N, T = config['shape']

        # normalization by the average power spectrum on the batch
        if x_observed is not None:
            sigma2_target = compute_sigma2(
                x_observed, config['J'], config['Q'],
                config['wav_type'], config['wav_norm'], config['high_freq'],
                config['rpad'], config['cuda'], config['nchunks']
            )
        else:
            sigma2_target = config['Rx'].query("c_type=='ps'")
        sigma2_target = sigma2_target.mean(0, keepdims=True)
        if sigma2_target.is_complex():
            raise ValueError("Normalization sigma2 should be real!.")

        # MODEL
        print("Initialize model")
        model = Model(T=T, N=N, Ns=None, channel_transforms=None,
                      norm_on_the_fly=False,
                      sigma2=sigma2_target, sigma2_L1=None, sigma2_L2=None, sigma2_Lphix=None,
                      estim_operator=None, A=None,
                      **config)
        if config['cuda']:
            model = model.cuda()
            if x_observed is not None:
                x_observed = x_observed.cuda()

        Rx_target = config['Rx']
        if Rx_target is None:
            print("Preparing target representation")
            Rx_target = model(x_observed)

        if model.all_coeff_types is not None:
            print(f"Model {config['model_type']} based on " +
                  f"{model.count_coefficients():,} statistics: ")
            for ctype in model.df.coeff_type.sort_values().unique():
                ncoeffs = model.count_coefficients(f"coeff_type=='{ctype}'")
                print(f" ---- {ctype} : {ncoeffs}")

        # OPTIM
        # initial time-series
        x0 = config['x0']
        if x0 is None:
            # infer correct x0 variance
            if x_observed is not None:
                x0_mean = to_numpy(x_observed).mean(-1, keepdims=True)
                x0_std = to_numpy(x_observed).std(-1, keepdims=True)
            else:
                x0_mean, x0_std = 0.0, 1.0

            def gen_wn(mean, std, shape):
                wn = np.random.randn(*shape)
                wn -= np.mean(wn, axis=-1, keepdims=True)
                wn /= np.std(wn, axis=-1, keepdims=True)

                return mean + std * wn

            x0 = gen_wn(x0_mean, x0_std, (B, N, T))

        # init loss, solver and convergence criterium
        loss = MSELossScat(J=config['J'],
                           histogram_moments=False,
                           wrap_avg=config['model_type'] == 'cov')
        solver = Solver(shape=torch.Size((B, N, T)), model=model, loss=loss,
                        Rx_target=Rx_target, x0=x0,
                        cuda=config['cuda'])
        check_conv_criterion = CheckConvCriterion(
            solver=solver, tol=config['tol_optim']
        )

        method, maxfun, jac = config['method'], config['maxfun'], config['jac']
        maxiter = config['max_iterations']

        tic = time()

        # Decide if the function provides gradient or not
        func = solver.joint if jac else solver.function
        try:
            res = scipy.optimize.minimize(
                func, x0.ravel(),
                method=method, jac=jac, callback=check_conv_criterion,
                options={
                    'ftol': 1e-24, 'gtol': 1e-24,
                    'maxiter': maxiter, 'maxfun': maxfun
                }
            )
            if res['nit'] == maxiter:
                # do not accept syntheses which haven't converged
                raise MaxIteration("MAX ITERATIONS REACHED. Optim failed.")
        except SmallEnoughException:  # raised by check_conv_criterion
            x_synt = check_conv_criterion.result
            it = check_conv_criterion.counter
            msg = "SmallEnoughException"

            if not isinstance(x_synt, np.ndarray):
                raise Exception("Something went wrong.")

            toc = time()

            flo, fgr = solver.joint(x_synt)
            flo, fgr = flo, np.max(np.abs(fgr))
            x_synt = x_synt.reshape(x0.shape)

            if not isinstance(msg, str):
                msg = msg.decode("ASCII")

            print(f"Optimization Exit Message : {msg}")
            print(f"found parameters in {toc - tic:0.2f}s, {it}" +
                  f" iterations -- {it / (toc - tic):0.2f}it/s")
            print(f"    abs sqrt error {flo ** 0.5:.2E}")
            print(f"    relative gradient error {fgr:.2E}")
            print(f"    loss0 {np.sqrt(solver.loss0):.2E}")

            if config['dtype'] == torch.float32:
                x_synt = x_synt.astype(np.float32)

            return x_synt  # (B, N, T)


def generate(x=None, Rx=None, S=1, shape=None,
             x0=None,
             model_type="scat_spectra", r=2, J=None, Q=1,
             wav_type='battle_lemarie', wav_norm='l1', high_freq=0.425,
             rpad=True,
             cross_params=None, diago_n=True,
             qs=None,
             coeff_types=None,
             max_iterations=10000, tol_optim=5e-4, seed=None,
             histogram_moments=False,
             skew_redundance=False,
             nchunks=1,
             cache_path=None, exp_name=None,
             cuda=False, gpus=None, num_workers=1):

    if x0 is not None and x0.ndim != 4:
        raise Exception("If provided, x0 should be of shape (B,nruns,N,T).")

    config = locals().copy()

    gpus = config['gpus']
    if gpus is not None and isinstance(gpus, str):
        gpus = [int(gpu) for gpu in gpus.split(',')]

    if config['shape'] is None:
        config['shape'] = config['x'].shape

    if x is not None:
        config['dtype'] = torch.float32 if x.dtype == np.float32 else torch.float64
    elif Rx is not None:
        config['dtype'] = Rx.y.dtype

    config['method'] = 'L-BFGS-B'
    config['maxfun'] = 2e6
    # origin of gradient, True: provided by solver, else estimated
    config['jac'] = True

    # batch size
    if x is not None:
        B = x.shape[0]
    elif shape is not None:
        B = shape[0]
    else:
        raise Exception("Should provide the shape of data to generate.")

    data_generator = ScatGenerator(B=B, **config)
    x_gen = data_generator.load(R=S, num_workers=num_workers)

    return x_gen


##################
# VIZUALIZE
##################

COLORS = ['skyblue', 'coral', 'lightgreen', 'darkgoldenrod', 'mediumpurple', 'red', 'purple', 'black',
          'paleturquoise'] + ['orchid'] * 20


def bootstrap_variance_complex(x, n_points, n_samples):
    """ Estimate variance of tensor x along last axis using bootstrap method. """
    # sample data uniformly
    sampling_idx = np.random.randint(
        low=0, high=x.shape[-1], size=(n_samples, n_points)
    )
    sampled_data = x[..., sampling_idx]

    # computes mean
    mean = sampled_data.mean(-1).mean(-1)

    # computes bootstrap variance
    var = (torch.abs(sampled_data.mean(-1) -
           mean[..., None]).pow(2.0)).sum(-1) / (n_samples - 1)

    return mean, var


def error_arg(z_mod, z_err, eps=1e-12):
    """ Transform an error on |z| into an error on Arg(z). """
    z_mod = np.maximum(z_mod, eps)
    return np.arctan(
        z_err / z_mod / np.sqrt(np.clip(1 - z_err ** 2 / z_mod ** 2, 1e-6, 1))
    )


def get_variance_of_average(z):
    """ Compute complex variance of the average (z1 + ... + zn) / n 
    assuming z1, ..., zn are iid (variance in the central limit theorem). """
    n = z.shape[0]
    var = torch.abs(z - z.mean(0, keepdim=True)).pow(2.0).sum(0).div(n-1)
    return var.div(n)


def plot_raw(Rx, ax=None, legend=True):
    """ Raw plot of the coefficients contained in Rx. 
        Colors indicate different coefficients type. 
        For each color, dark indicates real part, light indicates imag part. """

    if Rx.config['model_type'] is None:
        raise Exception(
            "Plotting scattering coefficients along time, not yet supported."
        )

    if "is_real" in Rx.df:
        columns_order = [
            c for c in Rx.df.columns if c not in ["coeff_type", "is_real"]
        ]
        columns_order = ["coeff_type", "is_real"] + columns_order
        Rx.df = Rx.df[columns_order]

    # average over the batch
    Rx = Rx.mean_batch()

    #
    if Rx.y.is_complex():
        Rx = format_to_real(Rx)
    Rx = Rx.sort()

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 2))
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, ctype in enumerate(Rx.df['coeff_type'].unique()):
        mask_real = np.where(Rx.df.eval(
            f"coeff_type=='{ctype}' & is_real==True"
        ))[0]
        mask_imag = np.where(Rx.df.eval(
            f"coeff_type=='{ctype}' & is_real==False"
        ))[0]
        ax.axvspan(mask_real.min(), mask_real.max(),
                   color=cycle[i], label=ctype if legend else None, alpha=0.65)
        if mask_imag.size > 0:
            ax.axvspan(mask_imag.min(), mask_imag.max(),
                       color=cycle[i], alpha=0.4)
        ax.axhline(0.0, color='black', linewidth=0.02)
    ax.plot(Rx.y[0, :, 0], linewidth=0.7, color='black',
            marker='+', markersize=1)
    if legend:
        ax.legend()
    return Rx.df


def plot_marginal_moments(Rxs, estim_bar=False,
                          axes=None, labels=None,
                          colors=None, linewidth=3.0, fontsize=30):
    """ Plot the marginal moments
        - (wavelet power spectrum) sigma^2(j)
        - (sparsity factors) s^2(j)

    :param Rxs: DescribedTensor or list of DescribedTensor
    :param estim_bar: display estimation error due to estimation on several realizations
    :param axes: custom axes: array of size 2
    :param labels: list of labels for each model output
    :param linewidth: curve linewidth
    :param fontsize: labels fontsize
    :return:
    """
    if isinstance(Rxs, DescribedTensor):
        Rxs = [Rxs]
    if labels is not None and len(Rxs) != len(labels):
        raise ValueError("Invalid number of labels")
    if axes is not None and axes.size != 2:
        raise ValueError(
            "The axes provided to plot_marginal_moments should be an array of size 2.")
    colors = colors or COLORS

    labels = labels or [''] * len(Rxs)
    axes = None if axes is None else axes.ravel()

    def plot_exponent(js, ax, label, color, y, y_err):
        plt.sca(ax)
        plt.plot(-js, y, label=label, linewidth=linewidth, color=color)
        if not estim_bar:
            plt.scatter(-js, y, marker='+', s=200,
                        linewidth=linewidth, color=color)
        else:
            eb = plt.errorbar(-js, y, yerr=y_err, capsize=4,
                              color=color, fmt=' ')
            eb[-1][0].set_linestyle('--')
        plt.yscale('log', base=2)
        plt.xlabel(r'$-j$', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks((-js).tolist(),
                   [fr'$-{j + 1}$' for j in js], fontsize=fontsize)

    if axes is None:
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        ax2 = plt.subplot2grid((1, 2), (0, 1))
        axes = [ax1, ax2]

    for i_lb, (lb, Rx) in enumerate(zip(labels, Rxs)):
        if 'coeff_type' not in Rx.df.columns:
            raise ValueError("The model output does not have the moments.")
        js = np.unique(Rx.query(is_low=False).df.jl1.dropna())

        has_power_spectrum = 'variance' in Rx.df.coeff_type.values
        has_sparsity = 'spars' in Rx.df.coeff_type.values

        # averaging on the logs may have strange behaviors because of the strict convexity of the log
        if has_power_spectrum:
            Wx2_nj = Rx.query(coeff_type=['variance'], rl=1, q=2.0, is_low=False).y[:, :, 0]
            if Wx2_nj.is_complex():
                Wx2_nj = Wx2_nj.real
            logWx2_n = torch.log2(Wx2_nj)

            # little subtlety here, we plot the log on the mean but the variance is the variance on the log
            logWx2_err = get_variance_of_average(logWx2_n) ** 0.5
            logWx2 = torch.log2(Wx2_nj.mean(0))
            # logWx2 -= logWx2[0].item()
            ps_norm_rectifier = 2 * 0.5 * np.arange(logWx2.shape[-1])
            plot_exponent(js, axes[0], lb, colors[i_lb],
                          2.0 ** (logWx2 + ps_norm_rectifier),
                          np.log(2) * logWx2_err * 2.0 ** logWx2)
            a, b = axes[0].get_ylim()
            avg = (logWx2 + ps_norm_rectifier).mean().item()
            a_min, b_max = 2 ** (avg - 2), 2 ** (avg + 2)
            if i_lb == len(labels) - 1:
                axes[0].set_ylim(min(a, a_min), max(b, b_max))
            if i_lb == len(labels) - 1 and any([lb != '' for lb in labels]):
                plt.legend(prop={'size': 15})
            plt.title(r'Variance $\Phi_2$', fontsize=fontsize)

        if has_sparsity:
            Wx1_nj = Rx.query(coeff_type=['spars'], rl=1, q=1.0, is_low=False).y[:, :, 0]
            if Wx1_nj.is_complex():
                Wx1_nj = Wx1_nj.real
            logWx1_nj = torch.log2(Wx1_nj)

            logWxs_n = 2 * logWx1_nj
            logWxs_err = get_variance_of_average(logWxs_n) ** 0.5
            logWxs = 2 * torch.log2(Wx1_nj.mean(0))
            plot_exponent(js, axes[1], lb, colors[i_lb],
                          2.0 ** logWxs,
                          np.log(2) * logWxs_err * 2.0 ** logWxs)
            a, b = axes[1].get_ylim()
            if i_lb == len(labels) - 1:
                axes[1].set_ylim(min(2 ** (-2), a), 1.0)
            if i_lb == len(labels) - 1 and any([lb != '' for lb in labels]):
                plt.legend(prop={'size': 15})
            plt.title(r'Sparsity $\Phi_1$', fontsize=fontsize)

    for ax in axes.ravel():
        ax.grid(True)


def plot_phase_envelope_spectrum(Rxs, estim_bar=False, self_simi_bar=False, theta_threshold=0.005,
                                 sigma2=None,
                                 axes=None, labels=None, colors=None, fontsize=30, single_plot=False, ylim=0.1):
    """ Plot the phase-envelope cross-spectrum C_{W|W|}(a) as two graphs : |C_{W|W|}| and Arg(C_{W|W|}).

    :param Rxs: DescribedTensor or list of DescribedTensor
    :param estim_bar: display estimation error due to estimation on several realizations
    :param self_simi_bar: display self-similarity error, it is a measure of scale regularity
    :param theta_threshold: rules phase instability
    :param sigma2: override normalization factors
    :param axes: custom axes: array of size 2
    :param labels: list of labels for each model output
    :param fontsize: labels fontsize
    :param single_plot: output all DescribedTensor on a single plot
    :param ylim: above y limit of modulus graph
    :return:
    """
    if isinstance(Rxs, DescribedTensor):
        Rxs = [Rxs]
    if labels is not None and len(Rxs) != len(labels):
        raise ValueError("Invalid number of labels")
    colors = colors or COLORS

    labels = labels or [''] * len(Rxs)
    columns = Rxs[0].df.columns
    J = Rxs[0].df.j.max() if 'j' in columns else Rxs[0].df.jl1.max()

    c_wmw = torch.zeros(len(labels), J-1, dtype=Rxs[0].y.dtype)
    err_estim = torch.zeros(len(labels), J-1)
    err_self_simi = torch.zeros(len(labels), J-1)

    for i_lb, Rx in enumerate(Rxs):

        if 'skewness' not in Rx.df.coeff_type.values:
            continue

        model_type = Rx.config['model_type']

        B = Rx.y.shape[0]

        for a in range(J):
            if model_type == 'inv_scat_spectra':
                c_mwm_n = Rx.query(coeff_type='skewness', a=a, is_low=False).y
                if c_mwm_n.shape[1] != 1:
                    raise Exception("ERROR. Should be selecting 1 "
                                    f"coefficient but got {c_mwm_n.shape[1]}")
                c_mwm_n = c_mwm_n[:, 0, 0]

                c_wmw[i_lb, a-1] = c_mwm_n.mean(0)
                err_estim[i_lb, a -
                          1] = get_variance_of_average(c_mwm_n).pow(0.5)
            else:
                c_mwm_nj = torch.zeros(B, J-a, dtype=Rx.y.dtype)
                for j1 in range(a, J):
                    coeff = Rx.query(
                        coeff_type='skewness', jl1=j1, jr1=j1-a, is_low=False
                    ).y
                    if coeff.shape[1] != 1:
                        raise Exception("ERROR. Should be selecting 1 "
                                        f"coefficient but got {coeff.shape[1]}")
                    c_mwm_nj[:, j1-a] = coeff[:, 0, 0]

                # the mean in j of the variance of time estimators
                c_wmw[i_lb, a-1] = c_mwm_nj.mean(0).mean(0)
                err_self_simi_n = (torch.abs(c_mwm_nj).pow(2.0).mean(1) - torch.abs(c_mwm_nj.mean(1)).pow(2.0)) / \
                    c_mwm_nj.shape[1]
                err_self_simi[i_lb, a-1] = err_self_simi_n.mean(0).pow(0.5)
                err_estim[i_lb, a -
                          1] = get_variance_of_average(c_mwm_nj.mean(1)).pow(0.5)

    c_wmw_mod, cwmw_arg = np.abs(c_wmw.numpy()), np.angle(c_wmw.numpy())
    err_self_simi, err_estim = to_numpy(err_self_simi), to_numpy(err_estim)
    err_self_simi_arg, err_estim_arg = error_arg(
        c_wmw_mod, err_self_simi), error_arg(c_wmw_mod, err_estim)

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
            plt.errorbar(a_s + plot_x_offset, y, yerr=y_err_self_simi,
                         capsize=4, color=color, fmt=' ')
        if estim_bar:
            plot_x_offset = 0.07 if self_simi_bar else 0.0
            eb = plt.errorbar(a_s + plot_x_offset, y,
                              yerr=y_err_estim, capsize=4, color=color, fmt=' ')
            eb[-1][0].set_linestyle('--')
        plt.title(r'Skewness $|\Phi_3|$', fontsize=fontsize)
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xticks(np.arange(1, J).tolist(),
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
            plt.errorbar(a_s + plot_x_offset, y, yerr=y_err_self_simi,
                         capsize=4, color=color, fmt=' ')
        if estim_bar:
            plot_x_offset = 0.07 if self_simi_bar else 0.0
            eb = plt.errorbar(a_s + plot_x_offset, y,
                              yerr=y_err_estim, capsize=4, color=color, fmt=' ')
            eb[-1][0].set_linestyle('--')
        plt.xticks(np.arange(1, J).tolist(), [
                   (rf'${j}$' if j % 2 == 1 else '') for j in np.arange(1, J)], fontsize=fontsize)
        plt.yticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi],
                   [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=fontsize)
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xlabel(r'$a$', fontsize=fontsize)
        plt.title(r'Skewness Arg $\Phi_3$', fontsize=fontsize)

    if axes is None:
        plt.figure(figsize=(5, 10) if single_plot else (len(labels) * 5, 10))
        ax_mod = plt.subplot2grid((2, 1), (0, 0))
        ax_mod.yaxis.set_tick_params(
            which='major', direction='in', width=1.5, length=7)
    else:
        plt.sca(axes[0])
    for i_lb, lb in enumerate(labels):
        plot_modulus(
            i_lb, lb, colors[i_lb], c_wmw_mod[i_lb], err_estim[i_lb], err_self_simi[i_lb])
        if i_lb == len(labels) - 1 and any([lb != '' for lb in labels]):
            plt.legend(prop={'size': 15})

    if axes is None:
        plt.subplot2grid((2, 1), (1, 0))
    else:
        plt.sca(axes[1])
    for i_lb, lb in enumerate(labels):
        plot_phase(lb, colors[i_lb], cwmw_arg[i_lb],
                   err_estim_arg[i_lb], err_self_simi_arg[i_lb])

    if axes is None:
        plt.tight_layout()

    for ax in axes.ravel():
        ax.grid(True)


def plot_scattering_spectrum(Rxs, estim_bar=False, self_simi_bar=False, bootstrap=True, theta_threshold=0.01,
                             sigma2=None,
                             axes=None, labels=None, colors=None, fontsize=40, ylim=2.0, d=1):
    """ Plot the scattering cross-spectrum C_S(a,b) as two graphs : |C_S| and Arg(C_S).

    :param Rxs: DescribedTensor or list of DescribedTensor
    :param estim_bar: display estimation error due to estimation on several realizations
    :param self_simi_bar: display self-similarity error, it is a measure of scale regularity
    :param theta_threshold: rules phase instability
    :param sigma2: override normalization factors
    :param axes: custom axes: array of size 2 x labels
    :param labels: list of labels for each model output
    :param fontsize: labels fontsize
    :param ylim: above y limit of modulus graph
    :return:
    """
    if isinstance(Rxs, DescribedTensor):
        Rxs = [Rxs]
    if labels is not None and len(Rxs) != len(labels):
        raise ValueError("Invalid number of labels.")
    if axes is not None and axes.size != 2 * len(Rxs):
        raise ValueError(
            f"Existing axes must be provided as an array of size {2 * len(Rxs)}")
    colors = colors or COLORS

    axes = None if axes is None else axes.reshape(2, len(Rxs))

    labels = labels or [''] * len(Rxs)
    i_graphs = np.arange(len(labels))

    columns = Rxs[0].df.columns
    J = Rxs[0].df.j.max() if 'j' in columns else Rxs[0].df.jl1.max()

    cs = torch.zeros(len(labels), J-1, J-1, dtype=Rxs[0].y.dtype)
    err_estim = torch.zeros(len(labels), J-1, J-1)
    err_self_simi = torch.zeros(len(labels), J-1, J-1)

    for i_lb, (Rx, lb, color) in enumerate(zip(Rxs, labels, colors)):

        if 'kurtosis' not in Rx.df.coeff_type.values:
            continue

        model_type = Rx.config['model_type']

        if self_simi_bar and model_type == 'inv_scat_spectra':
            raise ValueError("Impossible to output self-similarity error on "
                             "inv_scat_spectra model. Use a scat_spectra "
                             "model instead.")

        B = Rx.y.shape[0]

        for (a, b) in product(range(J-1), range(-J+1, 0)):
            if a - b >= J:
                continue

            # prepare covariances
            if model_type == "inv_scat_spectra":
                coeff_ab = Rx.query(
                    coeff_type='kurtosis', a=a, b=b, is_low=False
                ).y
                if coeff_ab.shape[1] != 1:
                    raise Exception(
                        "ERROR. Should be selecting 1 coefficient but got "
                        f"{coeff_ab.shape[1]}"
                    )
                coeff_ab = coeff_ab[:, 0, 0]
                cs[i_lb, a, J-1+b] = coeff_ab.mean(0)
            else:
                cs_nj = torch.zeros(B, J+b-a, dtype=Rx.y.dtype)
                for j1 in range(a, J+b):
                    coeff = Rx.query(
                        coeff_type='kurtosis', jl1=j1, jr1=j1-a, j2=j1-b,
                        is_low=False
                    ).y
                    if coeff.shape[1] != 1:
                        raise Exception(
                            "ERROR. Should be selecting 1 coefficient "
                            f"but got {coeff.shape[1]}"
                        )
                    cs_nj[:, j1 - a] = coeff[:, 0, 0]

                cs_j = cs_nj.mean(0)
                cs[i_lb, a, J-1+b] = cs_j.mean(0)
                if b == -J+a+1:
                    err_self_simi[i_lb, a, J-1+b] = 0.0
                else:
                    err_self_simi[i_lb, a, J-1+b] = torch.abs(cs_j - cs_j.mean(0, keepdim=True)) \
                        .pow(2.0).sum(0).div(J+b-a-1).pow(0.5)
                # compute estimation error
                if bootstrap:
                    # mean, var = bootstrap_variance_complex(cs_nj.transpose(0, 1), cs_nj.shape[0], 20000)
                    mean, var = bootstrap_variance_complex(
                        cs_nj.mean(1), cs_nj.shape[0], 20000)
                    err_estim[i_lb, a, J-1+b] = var.pow(0.5)
                else:
                    err_estim[i_lb, a, J-1+b] = (torch.abs(cs_nj).pow(2.0).mean(0) -
                                                 torch.abs(cs_nj.mean(0)).pow(2.0)) / (B - 1)

    cs, cs_mod, cs_arg = cs.numpy(), np.abs(cs.numpy()), np.angle(cs.numpy())
    err_self_simi, err_estim = to_numpy(err_self_simi), to_numpy(err_estim)
    err_self_simi_arg, err_estim_arg = error_arg(
        cs_mod, err_self_simi), error_arg(cs_mod, err_estim)

    # power spectrum normalization
    bs = np.arange(-J + 1, 0)[None, :] * d
    cs_mod /= (2.0 ** bs)
    err_self_simi /= (2.0 ** bs)
    err_estim /= (2.0 ** bs)

    # phase instability at z=0
    for z_arg in [cs_arg, err_self_simi_arg, err_estim_arg]:
        z_arg[cs_mod < theta_threshold] = 0.0

    def plot_modulus(label, y, y_err_estim, y_err_self_simi, title):
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
        plt.xticks(np.arange(-J + 1, 0).tolist(), [(rf'${b}$' if b % 2 == 1 else '') for b in np.arange(-J+1, 0)],
                   fontsize=fontsize)
        plt.xlabel(r'$b$', fontsize=fontsize)
        plt.ylim(-0.02, ylim)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.yticks(fontsize=fontsize)
        plt.locator_params(axis='x', nbins=J - 1)
        plt.locator_params(axis='y', nbins=5)
        if title:
            plt.title(r'Kurtosis $|\Phi_4|$', fontsize=fontsize)
        if label != '':
            plt.legend(prop={'size': 15})

    def plot_phase(y, y_err_estim, y_err_self_simi, title):
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
        plt.xticks(np.arange(-J+1, 0).tolist(), [(rf'${b}$' if b % 2 == 1 else '') for b in np.arange(-J+1, 0)],
                   fontsize=fontsize)
        plt.yticks(np.arange(-2, 3) * np.pi / 8,
                   [r'$-\frac{\pi}{4}$', r'$-\frac{\pi}{8}$',
                       r'$0$', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$'],
                   fontsize=fontsize)
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xlabel(r'$b$', fontsize=fontsize)
        if title:
            plt.title(r'Kurtosis Arg$\Phi_4$', fontsize=fontsize)

    if axes is None:
        plt.figure(figsize=(max(len(labels), 5) * 3, 10))
    for i_lb, lb in enumerate(labels):
        if axes is not None:
            plt.sca(axes[0, i_lb])
            ax_mod = axes[0, i_lb]
        else:
            ax_mod = plt.subplot2grid(
                (2, np.unique(i_graphs).size), (0, i_graphs[i_lb]))
        ax_mod.yaxis.set_tick_params(
            which='major', direction='in', width=1.5, length=7)
        ax_mod.yaxis.set_label_coords(-0.18, 0.5)
        plot_modulus(lb, cs_mod[i_lb], err_estim[i_lb],
                     err_self_simi[i_lb], i_lb == 0)

    for i_lb, lb in enumerate(labels):
        if axes is not None:
            plt.sca(axes[1, i_lb])
            ax_ph = axes[1, i_lb]
        else:
            ax_ph = plt.subplot2grid(
                (2, np.unique(i_graphs).size), (1, i_graphs[i_lb]))
        plot_phase(cs_arg[i_lb], err_estim_arg[i_lb],
                   err_self_simi_arg[i_lb], i_lb == 0)
        if i_lb == 0:
            ax_ph.yaxis.set_tick_params(
                which='major', direction='in', width=1.5, length=7)

    if axes is None:
        plt.tight_layout()
        leg = plt.legend(loc='upper center', ncol=1, fontsize=35, handlelength=1.0, labelspacing=1.0,
                         bbox_to_anchor=(1.3, 2.25, 0, 0))
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)

    for ax in axes.ravel():
        ax.grid(True)


def plot_dashboard(Rxs, estim_bar=False, self_simi_bar=False, bootstrap=True,
                   theta_threshold=[0.005, 0.1],
                   sigma2=None,
                   labels=None, colors=None,
                   linewidth=3.0, fontsize=20, ylim_phase=0.1, ylim_modulus=3.0,
                   figsize=None, axes=None):
    """ Plot the scattering covariance dashboard for multi-scale processes composed of:
        - (wavelet power spectrum) sigma^2(j)
        - (sparsity factors) s^2(j)
        - (phase-envelope cross-spectrum) C_{W|W|}(a) as two graphs : |C_{W|W|}| and Arg(C_{W|W|})
        - (scattering cross-spectrum) C_S(a,b) as two graphs : |C_S| and Arg(C_S)

    :param Rxs: DescribedTensor or list of DescribedTensor
    :param estim_bar: display estimation error due to estimation on several realizations
    :param self_simi_bar: display self-similarity error, it is a measure of scale regularity
    :param bootstrap: time variance computation method
    :param theta_threshold: rules phase instability
    :param sigma2: override normalization factors
    :param labels: list of labels for each model output
    :param linewidth: lines linewidth
    :param fontsize: labels fontsize
    :param ylim_phase: graph ylim for the phase
    :param ylim_modulus: graph ylim for the modulus
    :param figsize: figure size
    :param axes: custom array of axes, should be of shape (2, 2 + nb of representation to plot)
    :return:
    """
    if isinstance(Rxs, DescribedTensor):
        Rxs = [Rxs]
    for Rx in Rxs:
        if Rx.config is not None and Rx.config['model_type'] is None:
            raise Exception("Plotting of scattering coefficients along time "
                            "not yet implemented.")
        if 'nl' not in Rx.df.columns:
            Rx.df = make_description_compatible(Rx.df)
        ns_unique = Rx.df[['nl', 'nr']].dropna().drop_duplicates()
        if ns_unique.shape[0] > 1:
            raise ValueError("Plotting functions do not support multi-variate"
                             " representation other than single pair.")

    colors = colors or COLORS

    if axes is None:
        _, axes = plt.subplots(
            2, 2 + len(Rxs), figsize=figsize or (12+2*(len(Rxs)-1), 8))

    # marginal moments sigma^2 and s^2
    plot_marginal_moments(
        Rxs, estim_bar, axes[:, 0], labels, colors, linewidth, fontsize
    )

    # phase-envelope cross-spectrum
    plot_phase_envelope_spectrum(
        Rxs, estim_bar, self_simi_bar, theta_threshold[0], sigma2,
        axes[:, 1], labels, colors, fontsize, False, ylim_phase
    )

    # scattering cross spectrum
    plot_scattering_spectrum(
        Rxs, estim_bar, self_simi_bar, bootstrap, theta_threshold[1], sigma2,
        axes[:, 2:], labels, colors, fontsize, ylim_modulus
    )

    plt.tight_layout()

    return axes
