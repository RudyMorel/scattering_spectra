""" Frontend functions for analysis and generation. """
import os
from pathlib import Path
from itertools import product
from time import time

import scipy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from scatcov.utils import to_numpy, df_merge
from scatcov.data_source import ProcessDataLoader, FBmLoader, PoissonLoader, MRWLoader, SMRWLoader
from scatcov.scattering_network.scale_indexer import ScaleIndexer, ScatteringShape
from scatcov.scattering_network.layers_time import Wavelet
from scatcov.scattering_network.moments import ScatCoefficients, AvgLowPass, Cov, CovScaleInvariant
from scatcov.scattering_network.layers_basics import ChunkedModule, SkipConnection, Modulus, NormalizationLayer
from scatcov.scattering_network.described_tensor import Description, DescribedTensor
from scatcov.scattering_network.loss import MSELossScat
from scatcov.scattering_network.solver import Solver, CheckConvCriterion, SmallEnoughException

""" Notations

Dimension sizes:
- B: number of batches 
- R: number of realizations (used to estimate scattering covariance)
- N: number of data channels (N>1 : multivariate process)
- T: number of data samples (time samples)
- J: number of scales (octaves)
- Q: number of wavelets per octave
- r: number of conv layers in a scattering model

Tensor shapes:
- x: input, of shape  (B, N, T)
- Rx: output (DescribedTensor), of shape (B, K, T, 2) with K the number of coefficients in the representation
"""


##################
# DATA LOADING
##################

def load_data(process_name, R, T, cache_dir=None, **data_param):
    """ Time series data loading function.

    :param process_name: fbm, poisson, mrw, smrw, hawkes, turbulence or snp
    :param R: number of realizations
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
        cache_dir = Path(__file__).parents[0] / '_cached_dir'

    dtld = loader[process_name](cache_dir)
    x = dtld.load(R=R, n_files=R, T=T, **data_param).x

    return x


##################
# ANALYSIS
##################

def infer_description(model, model_type, N, sc_idxer, A):
    """ From a model get a dataframe that describes its output. """

    def pd_product_channel_single(df, Nl):
        """ Pandas cartesian product {(0,0), ..., (Nl-1, Nl-1))} x df """
        df_n = pd.DataFrame(np.stack([np.arange(Nl), np.arange(Nl)], 1), columns=['nl', 'nr'])

        return df_merge(df_n, df)

    if model_type is None:  # description of scattering module
        ns = pd.DataFrame(np.arange(N), columns=['n'])
        sc = pd.DataFrame(sc_idxer.get_all_idx(), columns=['sc'])
        js = pd.DataFrame(np.array([sc_idxer.idx_to_path(idx, squeeze=False) for idx in sc_idxer.get_all_idx()]),
                          columns=[f'j{o+1}' for o in range(sc_idxer.r_max)])
        rs = pd.DataFrame([sc_idxer.r(idx) for idx in sc_idxer.get_all_idx()], columns=['r'])
        low = pd.DataFrame([sc_idxer.is_low_pass(idx) for idx in sc_idxer.get_all_idx()], columns=['low'])
        sc_js = pd.concat([rs, sc, js, low], axis=1)
        a_s = pd.DataFrame(np.arange(A), columns=['al'])
        df = df_merge(ns, sc_js, a_s)

    elif model_type == 'scat':
        ns = pd.DataFrame(np.arange(N), columns=['nl'])
        sc = pd.DataFrame(sc_idxer.get_all_idx(), columns=['scl'])
        js = pd.DataFrame(np.array([sc_idxer.idx_to_path(idx, squeeze=False) for idx in sc_idxer.get_all_idx()]),
                          columns=[f'jl{o+1}' for o in range(sc_idxer.r_max)])
        sc_js = pd.concat([sc, js], axis=1)
        a_s = pd.DataFrame(np.arange(A), columns=['al'])
        qs = pd.DataFrame(model.module_marginal.qs.detach().cpu().numpy(), columns=['q'])
        df = df_merge(ns, sc_js, a_s, qs)
        df.replace(-1, np.nan, inplace=True)
        for o in range(1, sc_idxer.r_max):
            df[f'jl{o+1}'] = df[f'jl{o+1}'].astype('Int64')
        df['rl'] = [sc_idxer.r(sc) for sc in df['scl']]
        df['low'] = [sc_idxer.is_low_pass(sc) for sc in df['scl']]
        df['c_type'] = 'marginal'
        df = df.reindex(columns=['nl', 'rl', 'scl', 'jl1', 'j2', 'al', 'q', 'low', 'c_type'])

    elif model_type == 'cov':
        # Cov{Sx,Sx}
        df_q2 = model.module_cov.df_scale
        df_q2 = pd_product_channel_single(df_q2, N)

        # E{Sx}
        df_q1 = pd.DataFrame(columns=df_q2.columns)
        df_q1_orig = model.module_q1.df
        df_q1['scl'] = df_q1['scr'] = [sc_idxer.path_to_idx([j]) for j in df_q1_orig['j']]
        df_q1['rl'] = df_q1['rr'] = df_q1['q'] = 1
        df_q1['re'] = True
        df_q1['low'] = [sc_idxer.is_low_pass(sc) for sc in df_q1['scl']]
        df_q1['nr'] = df_q1['nl'] = df_q1_orig['n']
        df_q1['jr1'] = df_q1['jl1'] = df_q1_orig['j']
        df_q1['ar'] = df_q1['al'] = df_q1_orig['a']
        df_q1['j2'] = np.nan
        df_q1['j2'] = df_q1['j2'].astype('Int64')
        df_q1.loc[(df_q1.q == 1) & df_q1.low, ['c_type']] = 'mean'  # mean
        df_q1.loc[df_q1.q == 2, ['c_type']] = 'ps'  # power spectrum
        df_q1.loc[(df_q1.q == 1) & ~df_q1.low, ['c_type']] = 'spars'
        df = pd.concat([df_q1, df_q2])

    elif model_type == 'covreduced':
        # Cov{Sx,Sx} non invariant to scaling
        df_q2_orig = model.module_cov.df_scale.iloc[model.mask_noninv, :]
        df_q2 = df_q2_orig.copy()
        df_q2['a'] = df_q2_orig['jl1'] - df_q2_orig['jr1']
        df_q2['j'] = df_q2_orig['jl1']
        df_q2['b'] = pd.NA
        df_q2 = df_q2.drop(columns=['scl', 'scr', 'jl1', 'jr1', 'j2'])
        df_q2 = pd_product_channel_single(df_q2, N)
        # df_q2['j'] = pd.NA
        df_q2 = df_q2.reindex(columns=['nl', 'nr', 'q', 'rl', 'rr', 'j', 'a', 'b', 'al', 'ar', 're', 'low', 'c_type'])

        # Cov{Sx,Sx} invariant
        df_q2_inv = model.module_covinv.df_output
        df_q2_inv = pd_product_channel_single(df_q2_inv, N)
        df_q2_inv['j'] = pd.NA
        df_q2_inv = df_q2_inv.reindex(columns=['nl', 'nr', 'q', 'rl', 'rr', 'j', 'a', 'b', 'al', 'ar', 're', 'low', 'c_type'])

        # E{Sx}
        df_q1 = pd.DataFrame(columns=df_q2_inv.columns)
        df_q1_orig = model.module_q1.df
        df_q1['j'] = df_q1_orig['j']
        df_q1['rl'] = df_q1['rr'] = df_q1['q'] = 1
        df_q1['re'] = True
        df_q1['low'] = [sc_idxer.is_low_pass(j) for j in df_q1_orig['j']]
        df_q1['nr'] = df_q1['nl'] = df_q1_orig['n']
        df_q1['ar'] = df_q1['al'] = df_q1_orig['a']
        df_q1.loc[(df_q1.q == 1) & df_q1.low, ['c_type']] = 'mean'  # mean
        df_q1.loc[df_q1.q == 2, ['c_type']] = 'ps'  # power spectrum
        df_q1.loc[(df_q1.q == 1) & ~df_q1.low, ['c_type']] = 'spars'
        df = pd.concat([df_q1, df_q2, df_q2_inv])

    elif model_type == 'scat+cov':

        # Cov{Sx,Sx}
        df_q2 = model.module_cov.df_scale
        df_q2 = pd_product_channel_single(df_q2, N)

        # E{Sx}
        df_q1 = pd.DataFrame(columns=df_q2.columns)
        df_q1_orig = model.module_q1.df
        df_q1['scl'] = df_q1['scr'] = [sc_idxer.path_to_idx([j]) for j in df_q1_orig['j']]
        df_q1['rl'] = df_q1['rr'] = df_q1['q'] = 1
        df_q1['re'] = True
        df_q1['low'] = [sc_idxer.is_low_pass(sc) for sc in df_q1['scl']]
        df_q1['nr'] = df_q1['nl'] = df_q1_orig['n']
        df_q1['jr1'] = df_q1['jl1'] = df_q1_orig['j']
        df_q1['ar'] = df_q1['al'] = df_q1_orig['a']
        df_q1['j2'] = np.nan
        df_q1['j2'] = df_q1['j2'].astype('Int64')
        df_q1.loc[(df_q1.q == 1) & df_q1.low, ['c_type']] = 'mean'  # mean
        df_q1.loc[df_q1.q == 2, ['c_type']] = 'ps'  # power spectrum
        df_q1.loc[(df_q1.q == 1) & ~df_q1.low, ['c_type']] = 'spars'

        # E{|W|Wx||}
        ns = pd.DataFrame(np.arange(N), columns=['nl'])
        sc = pd.DataFrame([idx for idx in sc_idxer.get_all_idx() if sc_idxer.r(idx) == 2], columns=['scl'])
        js = pd.DataFrame(np.array([sc_idxer.idx_to_path(idx, squeeze=False) for idx in sc.values[:, 0]]),
                          columns=[f'j{o+1}' for o in range(sc_idxer.r_max)])
        sc_js = pd.concat([sc, js], axis=1)
        a_s = pd.DataFrame(np.arange(A), columns=['al'])
        df_scat2_q1 = pd.DataFrame(columns=df_q2.columns)
        df_scat2_q1[['nl', 'scl', 'jl1', 'j2', 'al']] = df_merge(ns, sc_js, a_s)
        df_scat2_q1['q'] = 1.0
        df_scat2_q1.replace(-1, np.nan, inplace=True)
        for o in range(1, sc_idxer.r_max):
            df_scat2_q1[f'j{o+1}'] = df_scat2_q1[f'j{o+1}'].astype('Int64')
        df_scat2_q1['rl'] = [sc_idxer.r(sc) for sc in df_scat2_q1['scl']]
        df_scat2_q1['low'] = [sc_idxer.is_low_pass(sc) for sc in df_scat2_q1['scl']]
        df_scat2_q1['c_type'] = 'scat2'

        df = pd.concat([df_q1, df_scat2_q1, df_q2])

    else:
        raise ValueError("Unrecognized model type.")

    return Description(df)


def init_model(B, N, T, J, Q1, Q2, r_max, wav_type1, wav_type2, high_freq, wav_norm,
               model_type, qs, sigma2, norm_on_the_fly,
               nchunks):
    """ Initialize a scattering covariance model.

    :param N: number of in_data channel
    :param T: number of time samples
    :param Q1: wavelets per octave first layer
    :param Q2: wavelets per octave second layer
    :param r_max: number convolution layers
    :param wav_type1: wavelet type for the first layer
    :param wav_type2: wavelet type for the second layer
    :param high_freq: central frequency of mother wavelet, 0.5 gives important aliasing
    :param wav_norm: wavelet normalization i.e. l1, l2
    :param model_type: moments to compute on scattering, ex: None, 'scat', 'cov', 'covreduced', 'scat+cov'
    :param qs: if model_type == 'marginal' the exponents of the scattering marginal moments
    :param sigma2: a tensor of size J, wavelet power spectrum to normalize the representation with
    :param norm_on_the_fly: normalize first wavelet layer on the fly
    :param keep_past: keep all layers in the output of the model
    :param nchunks: the number of chunks

    :return: a torch module
    """
    if nchunks < B:
        batch_chunk = nchunks
        cov_chunk = 1
    else:
        batch_chunk = B
        cov_chunk = nchunks // B

    # SCATTERING MODULE
    # time layers
    sc_idxer = ScaleIndexer(J=J, Qs=[Q1, Q2], r_max=2)
    W1 = Wavelet(T, J, Q1, wav_type1, high_freq, wav_norm, 1, sc_idxer)
    W2 = Wavelet(T, J, Q2, wav_type2, high_freq, wav_norm, 2, sc_idxer)

    # normalization layer
    if norm_on_the_fly:
        N_layer = NormalizationLayer(2, None, True)
    elif model_type == 'covreduced' or sigma2 is not None:
        N_layer = NormalizationLayer(2, sigma2.pow(0.5), False)
    else:
        N_layer = nn.Identity()

    if r_max == 1:
        scat_mod_l = [W1, N_layer]
    elif r_max == 2:
        scat_mod_l = [W1, N_layer, SkipConnection(nn.Sequential(Modulus(), W2), dim=2)]
    else:
        raise ValueError("More than 2 wavelet layers is not supported in the current version of the code.")
    scat_l = nn.Sequential(*scat_mod_l)

    shape_l = ScatteringShape(N, len(list(sc_idxer.get_all_idx())), 1, T)
    shape_r = shape_l

    # MODEL with MOMENTS on SCATTERING
    class Model(nn.Module):
        def __init__(self, model_type, scat_l, scat_r):
            super(Model, self).__init__()
            self.model_type = model_type
            self.scat_l = scat_l
            self.scat_r = scat_r
            self.module_marginal = ScatCoefficients(qs=qs or [1.0, 2.0])
            self.module_marginal1 = ScatCoefficients(qs=[1.0])
            self.module_q1 = AvgLowPass(N, 1, sc_idxer)
            self.module_cov = Cov(shape_l, shape_r, sc_idxer, cov_chunk) if (model_type is not None and 'cov' in model_type) else None
            self.module_covinv = CovScaleInvariant(shape_l, shape_r, sc_idxer, self.module_cov.df_scale) if model_type == 'covreduced' else None

            self.mask_noninv = self.module_cov.df_scale.where(low=True) | self.module_cov.df_scale.where(rl=1, rr=1) if model_type == 'covreduced' else None
            self.mask_scat2_q1 = torch.BoolTensor([sc_idxer.r(idx) == 2 for idx in sc_idxer.get_all_idx()])

        def forward(self, x):
            sxl = scat_l(x)
            sxr = sxl
            if self.model_type is None:
                repr = sxl.view((x.shape[0], -1, x.shape[-1]))
            elif self.model_type == 'scat':
                repr = self.module_marginal(sxl).view(x.shape[0], -1, 1)
            elif self.model_type == 'cov':
                m_q1 = self.module_q1(sxl)
                cov = self.module_cov(sxl, sxr).view(x.shape[0], -1, 1)
                repr = torch.cat([m_q1, cov], 1)
            elif self.model_type == 'covreduced':
                m_q1 = self.module_q1(sxl)
                cov = self.module_cov(sxl, sxr)
                covinv = self.module_covinv(cov).view(x.shape[0], -1, 1)  # invariant to scaling
                covnoninv = cov[:, :, :, self.mask_noninv, :].view(x.shape[0], -1, 1)
                repr = torch.cat([m_q1, covnoninv, covinv], 1)
            elif self.model_type == 'scat+cov':
                m_q1 = self.module_q1(sxl)
                cov = self.module_cov(sxl, sxr).view(x.shape[0], -1, 1)
                scat2_q1 = self.module_marginal1(sxl[:, :, self.mask_scat2_q1, :, :]).view(x.shape[0], -1, 1)
                repr = torch.cat([m_q1, scat2_q1, cov], 1)
            else:
                raise ValueError("Unrecognized model type.")
            return repr

    class DescribedModel(nn.Module):
        def __init__(self, module, description, model_type):
            super(DescribedModel, self).__init__()
            self.module = module
            self.description = description
            self.model_type = model_type
            self.c_types = ['mean', 'ps', 'spars', 'scat', 'phaseenv', 'envelope']

        def count_coefficients(self, **kwargs) -> int:
            """ Returns the number of moments satisfying kwargs. """
            return self.description.where(**kwargs).sum()

        def forward(self, x):
            return DescribedTensor(x=None, y=self.module(x), descri=self.description)

    model = Model(model_type, scat_l, None)
    description = infer_description(model, model_type, N, sc_idxer, 1)
    model = ChunkedModule(batch_chunk, model)
    model = DescribedModel(model, description, model_type)

    return model


def compute_sigma2(x, J, Q1, Q2, wav_type, high_freq, wav_norm, cuda):
    """ Computes power specturm sigma(j)^2 used to normalize scattering coefficients. """
    marginal_model = init_model(B=x.shape[0], N=x.shape[1], T=x.shape[-1], J=J, Q1=Q1, Q2=Q2, r_max=1,
                                wav_type1=wav_type, wav_type2=wav_type, high_freq=high_freq, wav_norm=wav_norm,
                                model_type='scat', qs=[2.0], sigma2=None, norm_on_the_fly=False,
                                nchunks=1)
    if cuda:
        x = x.cuda()
        marginal_model = marginal_model.cuda()

    sigma2 = marginal_model(x).y.reshape(x.shape[0], x.shape[1], -1)  # B x N x J

    return sigma2


def analyze(x, J=None, Q1=1, Q2=1,
            wav_type1='battle_lemarie', wav_type2='battle_lemarie', wav_norm='l1', high_freq=0.425,
            model_type='cov', qs=None, normalize=None, nchunks=1, cuda=False):
    """ Compute scattering based model.

    :param x: an array of shape (T, ) or (B, T) or (B, N, T)
    :param J: number of octaves
    :param Q1: number of scales per octave on first wavelet layer
    :param Q2: number of scales per octave on second wavelet layer
    :param wav_type1: wavelet type for the first layer
    :param wav_type2: wavelet type for the second layer
    :param wav_norm: wavelet normalization i.e. l1, l2
    :param high_freq: central frequency of mother wavelet, 0.5 gives important aliasing
    :param model_type: moments to compute on scattering
        None: compute Sx = W|Wx|(t,j1,j2) and keep time axis t
        "scat": compute marginal moments on Sx: E{|Sx|^q} by time average
        "cov": compute covariance on Sx: Cov{Sx, Sx} as well as E{|Wx|} and E{|Wx|^2} by time average
        "covreduced": compute reduced covariance: P Cov{Sx, Sx}, where P is the self-similar projection
        "scat+cov": both scat and cov
    :param normalize:
        None: no normalization,
        "each_ps": normalize Rx.y[b,:,:] by its power spectrum
        "batch_ps": normalize RX.y[b,:,:] by the average power spectrum over all trajectories b in the batch
    :param qs: exponent to use in a marginal model
    :param nchunks: nb of chunks, increase it to reduce memory usage
    :param cuda: does calculation on gpu

    :return: a DescribedTensor result
    """
    if normalize not in [None, "each_ps", "batch_ps"]:
        raise ValueError("Unrecognized normalization.")
    if model_type == 'covreduced' and normalize is None:
        raise ValueError("For covreduced model, user should provide a normalize argument.")
    if model_type is None and normalize is not None:
        raise ValueError("Can't use normalization with model_type=None.")

    if len(x.shape) == 1:  # assumes that x is of shape (T, )
        x = x[None, None, :]
    elif len(x.shape) == 2:  # assumes that x is of shape (B, T)
        x = x[:, None, :]

    B, N, T = x.shape
    x = torch.tensor(x)[:, :, None, None, :]

    if J is None:
        J = int(np.log2(T)) - 3

    # covreduced needs a spectrum normalization
    sigma2 = None
    if normalize is not None:
        sigma2 = compute_sigma2(x, J, Q1, Q2, wav_type1, high_freq, wav_norm, cuda)
        if normalize == "batch_ps":
            sigma2 = sigma2.mean(0, keepdim=True)

    # initialize model
    model = init_model(B=B, N=N, T=T, J=J, Q1=Q1, Q2=Q2, r_max=2,
                       wav_type1=wav_type1, wav_type2=wav_type2, high_freq=high_freq, wav_norm=wav_norm,
                       model_type=model_type, qs=qs, sigma2=sigma2, norm_on_the_fly=normalize=="each_ps",
                       nchunks=nchunks)

    # compute
    if cuda:
        x = x.cuda()
        model = model.cuda()

    Rx = model(x)

    if normalize is not None:
        # retrieve the power spectrum that was normalized
        mask_ps = Rx.descri.where(c_type='ps')
        if mask_ps.sum() != 0:
            Rx.y.real[:, mask_ps, :] = sigma2.reshape(sigma2.shape[0], -1, 1)

    return Rx.cpu().sort()


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
        B_target = kwargs['x'].shape[0]
        model_params = kwargs['model_params']
        N, T, J, Q1, Q2, r_max, wav_type, model_type = \
            (model_params[key] for key in ['N', 'T', 'J', 'Q1', 'Q2', 'r_max', 'wav_type1', 'model_type'])
        path_str = f"{self.model_name}_{wav_type}_B{B_target}_N{N}_T{T}_J{J}_Q1_{Q1}_Q2_{Q2}_rmax{r_max}_model_{model_type}" \
                   + f"_tol{kwargs['optim_params']['tol_optim']:.2e}" \
                   + f"_it{kwargs['optim_params']['it']}"
        return self.dir_name / path_str.replace('.', '_').replace('-', '_')

    def generate_trajectory(self, x, Rx, model_params, optim_params, gpu, dirpath):
        """ Performs cached generation. """
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        np.random.seed(None)
        filename = dirpath / str(np.random.randint(1e7, 1e8))
        if filename.is_file():
            raise OSError("File for saving this trajectory already exists.")

        x_torch = torch.tensor(x).unsqueeze(-2).unsqueeze(-2)

        # sigma = None
        sigma2 = compute_sigma2(x_torch, model_params['J'],
                                model_params['Q1'], model_params['Q2'],
                                model_params['wav_type1'], model_params['high_freq'], model_params['wav_norm'],
                                optim_params['cuda'])
        model_params['sigma2'] = sigma2.mean(0, keepdim=True)

        # prepare target representation
        if Rx is None:
            print("Preparing target representation")
            model_avg = init_model(B=x_torch.shape[0], **model_params)
            if gpu is not None:
                x_torch = x_torch.cuda()
                model_avg = model_avg.cuda()

            Rx = model_avg(x_torch).mean_batch().cpu()

        # prepare initial gaussian process
        x0_mean = x.mean(-1).mean(0)
        x0_var = np.var(x, axis=-1).mean(0)

        def gen_wn(shape, mean, var):
            wn = np.random.randn(*shape)
            wn -= wn.mean(axis=-1, keepdims=True)
            wn /= np.std(wn, axis=-1, keepdims=True)

            return wn * var + mean

        x0 = gen_wn(x.shape, x0_mean, x0_var ** 0.5)

        # init model
        print("Initialize model")
        model = init_model(B=x_torch.shape[0], **model_params)

        # init loss, solver and convergence criterium
        loss = MSELossScat()
        solver_fn = Solver(model=model, loss=loss, xf=x, Rxf=Rx, x0=x0,
                           weights=None, cuda=optim_params['cuda'], relative_optim=optim_params['relative_optim'])

        check_conv_criterion = CheckConvCriterion(solver=solver_fn, tol=optim_params['tol_optim'])

        print('Embedding: uses {} coefficients {}'.format(
            model.count_coefficients(),
            ' '.join(
                ['{}={}'.format(c_type, model.count_coefficients(c_type=c_type))
                 for c_type in model.c_types])
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
        flo, fgr = flo, np.max(np.abs(fgr))
        x_synt = x_opt.reshape(x0.shape)

        if not isinstance(msg, str):
            msg = msg.decode("ASCII")

        print('Optimization Exit Message : ' + msg)
        print(f"found parameters in {toc - tic:0.2f}s, {it} iterations -- {it / (toc - tic):0.2f}it/s")
        print(f"    abs sqrt error {flo ** 0.5:.2E}")
        print(f"    relative gradient error {fgr:.2E}")
        print(f"    loss0 {solver_fn.loss0:.2E}")

        return x_synt  # S x N x T

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
            x = self.generate_trajectory(**kwargs)
            fname = f"{np.random.randint(1e7, 1e8)}.npy"
            np.save(str(kwargs['dirpath'] / fname), x)
        except ValueError as e:
            print(e)
            return


def generate(x, Rx=None, S=1, J=None, Q1=1, Q2=1, wav_type='battle_lemarie', wav_norm='l1', high_freq=0.425,
             model_type='cov', qs=None, nchunks=1, it=10000, tol_optim=5e-4,
             generated_dir=None, exp_name=None, cuda=False, gpus=None, num_workers=1):
    """ Generate new realizations of x from a scattering covariance model.
    We first compute the scattering covariance representation of x and then sample it using gradient descent.

    :param x: an array of shape (T, ) or (B, T) or (B, N, T)
    :param Rx: instead of x, the representation to generate from
    :param J: number of octaves
    :param Q1: number of scales per octave on first wavelet layer
    :param Q2: number of scales per octave on second wavelet layer
    :param wav_type: wavelet type
    :param wav_norm: wavelet normalization i.e. l1, l2
    :param high_freq: central frequency of mother wavelet, 0.5 gives important aliasing
    :param model_type: moments to compute on scattering, ex: None, 'scat', 'cov', 'covreduced'
    :param qs: if model_type == 'marginal' the exponents of the scattering marginal moments
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
    if len(x.shape) == 1:  # assumes that x is of shape (T, )
        x = x[None, None, :]
    elif len(x.shape) == 2:  # assumes that x is of shape (B, T)
        x = x[:, None, :]

    B, N, T = x.shape

    if J is None:
        J = int(np.log2(T)) - 5
    if qs is None:
        qs = [1.0, 2.0]
    if generated_dir is None:
        generated_dir = Path(__file__).parents[0] / '_cached_dir'

    # use a GenDataLoader to cache trajectories
    dtld = GenDataLoader(exp_name or 'gen_scat_cov', generated_dir, num_workers)

    # MODEL params
    model_params = {
        'N': N, 'T': T, 'J': J, 'Q1': Q1, 'Q2': Q2, 'r_max': 2,
        'wav_type1': wav_type,  'wav_type2': wav_type,  # 'battle_lemarie' 'morlet' 'shannon'
        'high_freq': high_freq,  # 0.323645 or 0.425
        'wav_norm': wav_norm,
        'model_type': model_type, 'qs': qs, 'sigma2': None, 'norm_on_the_fly': False,
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
        'tol_optim': tol_optim
    }

    # multi-processed generation
    x_gen = dtld.load(R=S, n_files=int(np.ceil(S/B)), x=x, Rx=Rx, model_params=model_params, optim_params=optim_params).x

    return x_gen


##################
# VIZUALIZE
##################

COLORS = ['skyblue', 'coral', 'lightgreen', 'darkgoldenrod', 'mediumpurple', 'red', 'purple', 'black',
          'paleturquoise'] + ['orchid'] * 20


def bootstrap_variance_complex(x, n_points, n_samples):
    """ Estimate variance of tensor x along last axis using bootstrap method. """
    # sample data uniformly
    sampling_idx = np.random.randint(low=0, high=x.shape[-1], size=(n_samples, n_points))
    sampled_data = x[..., sampling_idx]

    # computes mean
    mean = sampled_data.mean(-1).mean(-1)

    # computes bootstrap variance
    var = (torch.abs(sampled_data.mean(-1) - mean[..., None]).pow(2.0)).sum(-1) / (n_samples - 1)

    return mean, var


def error_arg(z_mod, z_err, eps=1e-12):
    """ Transform an error on |z| into an error on Arg(z). """
    z_mod = np.maximum(z_mod, eps)
    return np.arctan(z_err / z_mod / np.sqrt(np.clip(1 - z_err ** 2 / z_mod ** 2, 1e-6, 1)))


def get_variance(z):
    """ Compute complex variance of a sequence of complex numbers z1, z2, ... """
    B = z.shape[0]
    return torch.abs(z - z.mean(0, keepdim=True)).pow(2.0).sum(0).div(B-1).div(B)


def plot_marginal_moments(Rxs, estim_bar=False,
                          axes=None, labels=None, linewidth=3.0, fontsize=30):
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
        raise ValueError("The axes provided to plot_marginal_moments should be an array of size 2.")

    labels = labels or [''] * len(Rxs)
    axes = None if axes is None else axes.ravel()

    def get_data(Rx, q):
        Wx_nj = Rx.select(rl=1, c_type=['ps', 'spars', 'marginal'], q=q, low=False)[:, :, 0]
        if Wx_nj.dtype == torch.complex128:
            Wx_nj = Wx_nj.real
        logWx_nj = torch.log2(Wx_nj)
        return logWx_nj

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

    for i_lb, (lb, Rx) in enumerate(zip(labels, Rxs)):
        if 'c_type' not in Rx.descri.columns:
            raise ValueError("The model output does not have the moments.")
        columns = Rx.descri.columns
        js = np.unique(Rx.descri.reduce(rl=1, low=False).j if 'j' in columns else Rx.descri.reduce(low=False).jl1)

        has_power_spectrum = 2.0 in Rx.descri.q.values
        has_sparsity = 1.0 in Rx.descri.q.values

        # averaging on the logs may have strange behaviors because of the strict convexity of the log
        # if you prefer to look at the log of the mean, then to a .mean_batch() on the representation before ploting it
        if has_power_spectrum:
            logWx2_n = get_data(Rx, 2.0)
            logWx2_err = get_variance(logWx2_n) ** 0.5
            logWx2 = logWx2_n.mean(0)
            logWx2 -= logWx2[-1].item()
            plot_exponent(js, 0, axes[0], lb, COLORS[i_lb], 2.0 ** logWx2, np.log(2) * logWx2_err * 2.0 ** logWx2)
            if i_lb == len(labels) - 1 and any([lb != '' for lb in labels]):
                plt.legend(prop={'size': 15})
            plt.title('Wavelet Spectrum', fontsize=fontsize)

            if has_sparsity:
                logWx1_n = get_data(Rx, 1.0)
                logWxs_n = 2 * logWx1_n - logWx2_n.mean(0, keepdims=True)
                logWxs_err = get_variance(logWxs_n) ** 0.5
                logWxs = logWxs_n.mean(0)
                plot_exponent(js, 1, axes[1], lb, COLORS[i_lb], 2.0 ** logWxs, np.log(2) * logWxs_err * 2.0 ** logWxs)
                if i_lb == len(labels) - 1 and any([lb != '' for lb in labels]):
                    plt.legend(prop={'size': 15})
                plt.title('Sparsity factor', fontsize=fontsize)


def plot_phase_envelope_spectrum(Rxs, estim_bar=False, self_simi_bar=False, theta_threshold=0.005,
                                 axes=None, labels=None, fontsize=30, single_plot=False, ylim=0.09, title=True):
    """ Plot the phase-envelope cross-spectrum C_{W|W|}(a) as two graphs : |C_{W|W|}| and Arg(C_{W|W|}).

    :param Rxs: DescribedTensor or list of DescribedTensor
    :param estim_bar: display estimation error due to estimation on several realizations
    :param self_simi_bar: display self-similarity error, it is a measure of scale regularity
    :param theta_threshold: rules phase instability
    :param axes: custom axes: array of size 2
    :param labels: list of labels for each model output
    :param fontsize: labels fontsize
    :param single_plot: output all DescribedTensor on a single plot
    :param ylim: above y limit of modulus graph
    :param title: put title on each graph
    :return:
    """
    if isinstance(Rxs, DescribedTensor):
        Rxs = [Rxs]
    if labels is not None and len(Rxs) != len(labels):
        raise ValueError("Invalid number of labels")

    labels = labels or [''] * len(Rxs)
    columns = Rxs[0].descri.columns
    J = Rxs[0].descri.j.max() if 'j' in columns else Rxs[0].descri.jl1.max()

    c_wmw = torch.zeros(len(labels), J-1, dtype=torch.complex128)
    err_estim = torch.zeros(len(labels), J-1)
    err_self_simi = torch.zeros(len(labels), J-1)

    for i_lb, Rx in enumerate(Rxs):
        # infer model type
        if ('jl1' in Rx.descri) and ('jr1' in Rx.descri):
            model_type = 'cov'
        elif 'j' in Rx.descri:
            model_type = 'covreduced'
        else:
            continue

        B = Rx.y.shape[0]

        sigma2 = Rx.select(rl=1, rr=1, q=2, low=False).real.mean(0, keepdim=True)[:, :, 0]

        for a in range(1, J):
            if model_type == 'covreduced':
                c_mwm_n = Rx.select(c_type='phaseenv', a=a, low=False)
                assert c_mwm_n.shape[1] == 1, f"ERROR. Should be selecting 1 coefficient but got {c_mwm_n.shape[1]}"
                c_mwm_n = c_mwm_n[:, 0, 0]

                c_wmw[i_lb, a-1] = c_mwm_n.mean(0)
                err_estim[i_lb, a-1] = get_variance(c_mwm_n).pow(0.5)
            else:
                c_mwm_nj = torch.zeros(B, J-a, dtype=torch.complex128)
                for j1 in range(a, J):
                    coeff = Rx.select(c_type='phaseenv', jl1=j1, jr1=j1-a, low=False)
                    coeff = coeff[:, 0, 0]
                    coeff /= sigma2[:, j1, ...].pow(0.5) * sigma2[:, j1-a, ...].pow(0.5)
                    c_mwm_nj[:, j1-a] = coeff

                # the mean in j of the variance of time estimators
                c_wmw[i_lb, a-1] = c_mwm_nj.mean(0).mean(0)
                err_self_simi_n = (torch.abs(c_mwm_nj).pow(2.0).mean(1) - torch.abs(c_mwm_nj.mean(1)).pow(2.0)) / c_mwm_nj.shape[1]
                err_self_simi[i_lb, a-1] = err_self_simi_n.mean(0).pow(0.5)
                err_estim[i_lb, a-1] = get_variance(c_mwm_nj.mean(1)).pow(0.5)

    c_wmw_mod, cwmw_arg = np.abs(c_wmw.numpy()), np.angle(c_wmw.numpy())
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


def plot_scattering_spectrum(Rxs, estim_bar=False, self_simi_bar=False, bootstrap=True, theta_threshold=0.01,
                             axes=None, labels=None, fontsize=40, ylim=2.0, d=1):
    """ Plot the scattering cross-spectrum C_S(a,b) as two graphs : |C_S| and Arg(C_S).

    :param Rxs: DescribedTensor or list of DescribedTensor
    :param estim_bar: display estimation error due to estimation on several realizations
    :param self_simi_bar: display self-similarity error, it is a measure of scale regularity
    :param theta_threshold: rules phase instability
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
        raise ValueError(f"Existing axes must be provided as an array of size {2 * len(Rxs)}")

    axes = None if axes is None else axes.reshape(2, len(Rxs))

    labels = labels or [''] * len(Rxs)
    i_graphs = np.arange(len(labels))

    columns = Rxs[0].descri.columns
    J = Rxs[0].descri.j.max() if 'j' in columns else Rxs[0].descri.jl1.max()

    cs = torch.zeros(len(labels), J-1, J-1, dtype=torch.complex128)
    err_estim = torch.zeros(len(labels), J-1, J-1)
    err_self_simi = torch.zeros(len(labels), J-1, J-1)

    for i_lb, (Rx, lb, color) in enumerate(zip(Rxs, labels, COLORS)):
        # infer model type
        if ('jl1' in Rx.descri) and ('jr1' in Rx.descri):
            model_type = 'cov'
        elif 'j' in Rx.descri:
            model_type = 'covreduced'
        else:
            continue

        if self_simi_bar and model_type == 'covreduced':
            raise ValueError("Impossible to output self-similarity error on covreduced model. Use a cov model instead.")

        B = Rx.y.shape[0]

        sigma2 = Rx.select(rl=1, rr=1, q=2, low=False).real.mean(0, keepdim=True)[:, :, 0]

        for (a, b) in product(range(J-1), range(-J+1, 0)):
            if a - b >= J:
                continue

            # prepare covariances
            if model_type == 'cov':
                cs_nj = torch.zeros(B, J+b-a, dtype=torch.complex128)
                for j1 in range(a, J+b):
                    coeff = Rx.select(c_type='envelope', jl1=j1, jr1=j1-a, j2=j1-b, low=False)
                    coeff = coeff[:, 0, 0]
                    coeff /= sigma2[:, j1, ...].pow(0.5) * sigma2[:, j1 - a, ...].pow(0.5)
                    cs_nj[:, j1 - a] = coeff

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
                    mean, var = bootstrap_variance_complex(cs_nj.mean(1), cs_nj.shape[0], 20000)
                    err_estim[i_lb, a, J-1+b] = var.pow(0.5)
                else:
                    err_estim[i_lb, a, J-1+b] = (torch.abs(cs_nj).pow(2.0).mean(0) -
                                                 torch.abs(cs_nj.mean(0)).pow(2.0)) / (B - 1)
            else:
                coeff_ab = Rx.select(c_type='envelope', a=a, b=b, low=False)
                coeff_ab = coeff_ab[:, 0, 0]
                cs[i_lb, a, J-1+b] = coeff_ab.mean(0)

    cs, cs_mod, cs_arg = cs.numpy(), np.abs(cs.numpy()), np.angle(cs.numpy())
    err_self_simi, err_estim = to_numpy(err_self_simi), to_numpy(err_estim)
    err_self_simi_arg, err_estim_arg = error_arg(cs_mod, err_self_simi), error_arg(cs_mod, err_estim)

    # power spectrum normalization
    bs = np.arange(-J + 1, 0)[None, :] * d
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


def plot_dashboard(Rxs, estim_bar=False, self_simi_bar=False, bootstrap=True, theta_threshold=None,
                   labels=None, linewidth=3.0, fontsize=20, ylim_phase=0.09, ylim_modulus=2.0, figsize=None, axes=None):
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
    if isinstance(Rxs, DescribedTensor):
        Rxs = [Rxs]
    for Rx in Rxs:
        if 'nr' in Rx.descri.columns:
            ns = Rx.descri.to_array(['nl', 'nr'])
        else:
            ns = Rx.descri.to_array(['nl'])
        ns_unique = set([tuple(n) for n in ns])
        if len(ns_unique) != 1:
            raise ValueError("Plotting functions do not support multi-variate representation other than "
                             "univariate or single pair.")

    if axes is None:
        _, axes = plt.subplots(2, 2 + len(Rxs), figsize=figsize or (10 + 5 * (len(Rxs) - 1), 10))

    # marginal moments sigma^2 and s^2
    plot_marginal_moments(Rxs, estim_bar, axes[:, 0], labels, linewidth, fontsize)

    # phase-envelope cross-spectrum
    plot_phase_envelope_spectrum(Rxs, estim_bar, self_simi_bar, theta_threshold[0], axes[:, 1], labels, fontsize, False, ylim_phase)

    # scattering cross spectrum
    plot_scattering_spectrum(Rxs, estim_bar, self_simi_bar, bootstrap, theta_threshold[1], axes[:, 2:], labels, fontsize, ylim_modulus)

    plt.tight_layout()
