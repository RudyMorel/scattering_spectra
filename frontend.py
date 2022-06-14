import os
from pathlib import Path
import numpy as np

import utils.complex_utils as cplx
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

    print(X.device)

    RX = model(X)

    return RX.cpu()


##################
# SYNTHESIS
##################
