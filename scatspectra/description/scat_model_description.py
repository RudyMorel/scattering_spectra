from itertools import product
import numpy as np
import pandas as pd

from scatspectra.utils import (
    df_product, df_product_channel_single, df_product_channel_double
)
from scatspectra.description import ScaleIndexer


def scattering_network_description(
    r: int,
    N_out: int,
    scale_indexer: ScaleIndexer
) -> pd.DataFrame:
    """ Assemble description of output of Sx = (Wx, W|Wx|, ..., W|...|Wx||). """

    r_max = scale_indexer.r
    sc_idces = scale_indexer.sc_idces[r-1]

    # channels
    ns = pd.DataFrame(np.arange(N_out), columns=['n'])

    # scale indices
    scs = pd.DataFrame(sc_idces, columns=['sc'])

    # scale paths (j1, ..., jr)
    js_arr = np.array([
        scale_indexer.idx_to_path(sc, squeeze=False) for sc in scs.sc.values
    ])
    js = pd.DataFrame(js_arr, columns=[f'j{r}' for r in range(1, r_max+1)])
    scs_js = pd.concat([scs, js], axis=1)

    # phases
    a_s = pd.DataFrame(np.arange(1), columns=['a'])

    df = df_product(ns, scs_js, a_s)

    df['is_low'] = [scale_indexer.is_low_pass(sc) for sc in df['sc'].values]
    df['r'] = [scale_indexer.order(sc) for sc in df['sc'].values]

    output_columns = [
        'r', 'n', 'sc', *[f'j{r}' for r in range(1, r_max + 1)], 'a', 'is_low'
    ]
    return df.reindex(columns=output_columns)


def scattering_coefficients_description(
    N: int,
    sc_idxer: ScaleIndexer,
    qs: np.ndarray
) -> pd.DataFrame:

    r_max = sc_idxer.r

    df = pd.concat([
        scattering_network_description(r, N, sc_idxer)
        for r in range(1, r_max+1)
    ])
    df['coeff_type'] = "scat_marginal"
    df['is_real'] = True
    qs = pd.DataFrame(qs, columns=['q'])  # type: ignore
    df = df_product(df, qs)  # type: ignore

    return df


def make_description_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """ Convert marginal description to correlation description. """
    df = df.rename(columns={'r': 'rl', 'n': 'nl',
                            'sc': 'scl', 'j1': 'jl1', 'a': 'al'})
    df['is_real'] = True
    df['nr'] = df['nl']
    df['rr'] = df['scr'] = df['ar'] = df['jr1'] = pd.NA
    df['coeff_type'] = ['mean' if low else 'spars' for low in df.is_low.values]
    df = df.reindex(columns=['coeff_type', 'nl', 'nr', 'q', 'rl', 'rr',
                             'scl', 'scr', 'jl1', 'jr1', 'j2', 'al', 'ar', 'is_real', 'is_low'])

    return df


def build_description_mean_spars(N: int, sc_idxer: ScaleIndexer) -> pd.DataFrame:
    """ Assemble the description of averages E{Wx} and E{|Wx|}. """
    df = scattering_network_description(1, N, sc_idxer)
    df = df.query("r==1")

    # compatibility with covariance description
    df = make_description_compatible(df)
    df['q'] = 1

    return df


def create_scale_description(
    scls: np.ndarray,
    scrs: np.ndarray,
    sc_idxer: ScaleIndexer
) -> pd.DataFrame:
    """ Return the dataframe that describes the scale association in the output of forward. """
    info_l = []
    for (scl, scr) in product(scls, scrs):
        rl, rr = sc_idxer.order(scl), sc_idxer.order(scr)
        ql, qr = sc_idxer.Q[rl-1], sc_idxer.Q[rr-1]
        if rl > rr:
            continue

        pathl, pathr = sc_idxer.idx_to_path(scl), sc_idxer.idx_to_path(scr)
        jl, jr = sc_idxer.idx_to_path(
            scl, squeeze=False), sc_idxer.idx_to_path(scr, squeeze=False)

        if rl == rr == 2 and pathl < pathr:
            continue

        # remove scale paths that has the redundancy |j1|j1 for capturing envelope correlations
        if rl == rr == 2 and np.any((np.diff(pathl) == 0) | (np.diff(pathr) == 0)):
            continue

        # correlate low pass with low pass or band pass with band pass and nothing else
        if (sc_idxer.is_low_pass(scl) and not sc_idxer.is_low_pass(scr)) or \
                (not sc_idxer.is_low_pass(scl) and sc_idxer.is_low_pass(scr)):
            continue

        # only consider wavelets with non-negligibale overlapping support in Fourier
        # weak condition: last wavelets must be closer than one octave
        # if abs(pathl[-1] / ql - pathr[-1] / qr) >= 1:
        #     continue
        # strong condition: last wavelets must be equal
        if abs(pathl[-1] / ql - pathr[-1] / qr) > 0:
            continue

        low = sc_idxer.is_low_pass(scl)

        info_l.append(('variance' if rl * rr == 1 else 'skewness' if rl * rr == 2 else 'kurtosis',
                       2, rl, rr, scl, scr, *jl, *jr, 0, 0, low or scl == scr, low))

    out_columns = ['coeff_type', 'q', 'rl', 'rr', 'scl', 'scr'] + \
        [f'jl{r}' for r in range(1, sc_idxer.r + 1)] + \
        [f'jr{r}' for r in range(1, sc_idxer.r + 1)] + \
        ['al', 'ar', 'is_real', 'is_low']
    df_scale = pd.DataFrame(info_l, columns=out_columns)

    # now do a diagonal or cartesian product along channels
    df_scale = (
        df_scale
        .drop('jl2', axis=1)
        .rename(columns={'jr2': 'j2'})
    )

    return df_scale


def build_description_correlation(N: int, sc_idxer: ScaleIndexer, multivariate: bool) -> pd.DataFrame:
    """ Assemble the description the phase modulus correlation E{Sx, Sx}. """
    scs_r1, scs_r2 = sc_idxer.sc_idces[:2]

    df_ww = create_scale_description(scs_r1, scs_r1, sc_idxer)
    df_wmw = create_scale_description(scs_r1, scs_r2, sc_idxer)
    df_mw = create_scale_description(scs_r2, scs_r2, sc_idxer)

    def channel_expand(df, N1, N2):
        if multivariate:
            return df_product_channel_double(df, N1, N2)
        return df_product_channel_single(df, N1, "same")

    df_cov = pd.concat([
        channel_expand(df_ww, N, N),
        channel_expand(df_wmw, N, N),
        channel_expand(df_mw, N, N)
    ])

    return df_cov


def create_scale_invariant_description(sc_idxer: ScaleIndexer) -> pd.DataFrame:
    """ Return the dataframe that describes the output of forward. """
    J = sc_idxer.JQ(1)  # TODO: this implies Q=1

    data = []

    # skewness coefficients <Wx(t,j) W|Wx|(t,j-a,j)^*>_t
    for a in range(int(sc_idxer.strictly_increasing), J):
        data.append((2, 1, 2, a, pd.NA, 0, 0, False, False, 'skewness'))

    # kurtosis coefficients <W|Wx|(t,j,j-b) W|Wx|(t,j-a,j-b)^*>_t
    for (a, b) in product(range(J-1), range(-J+1, 0)):
        if a - b >= J:
            continue
        data.append((2, 2, 2, a, b, 0, 0, a == 0, False, 'kurtosis'))

    df = pd.DataFrame(
        data,
        columns=['q', 'rl', 'rr', 'a', 'b', 'al', 'ar',
                 'is_real', 'is_low', 'coeff_type']
    )

    return df


def build_description_histograms(sc_idxer: ScaleIndexer) -> pd.DataFrame:
    """ Assemble the description of the histograms of scattering coefficients. """
    data = []
    # skewness moments P(\delta_j x (t) > 0)
    for j in range(0, sc_idxer.JQ(1)):
        data.append((
            'hist_shewness', 0, pd.NA, 0, 0, pd.NA, j, pd.NA, j, pd.NA, pd.NA, 0, pd.NA, True, j==sc_idxer.JQ(1)
        ))
    # low-moment kurtosis E{|\delta_j x|}
    for j in range(0, sc_idxer.JQ(1)):
        data.append((
            'hist_kurtosis', 0, pd.NA, 1, 0, pd.NA, j, pd.NA, j, pd.NA, pd.NA, 0, pd.NA, True, j==sc_idxer.JQ(1)
        ))
    # log-envelope energy, E{ |log|Wx||^2 }
    for j in range(0, sc_idxer.JQ(1)+1):
        data.append((
            'hist_variance_log', 0, pd.NA, 2, 0, pd.NA, j, pd.NA, j, pd.NA, pd.NA, 0, pd.NA, True, j==sc_idxer.JQ(1)
        ))
    return pd.DataFrame(
        data,
        columns=['coeff_type', 'nl', 'nr', 'q', 'rl', 'rr', 'scl', 'scr', 'jl1', 'jr1', 'j2', 'al', 'ar', 'is_real', 'is_low']
    )