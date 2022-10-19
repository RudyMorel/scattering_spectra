""" Moments to be used on top of a scattering transform. """
from typing import *
from itertools import product
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scatcov.utils import df_product
from scatcov.scattering_network.scale_indexer import ScaleIndexer
from scatcov.scattering_network.described_tensor import Description


class ScatCoefficients(nn.Module):
    """ Compute per channel (marginal) order q moments. """
    def __init__(self, qs: List[float]):
        super(ScatCoefficients, self).__init__()
        self.c_types = ['marginal']

        self.register_buffer('qs', torch.tensor(qs))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """ Computes E[|Sx|^q].

        :param x: B x N x js x A x T tensor
        :return: B x N x js x 1 x len(qs) tensor
        """
        return (torch.abs(x).unsqueeze(-1) ** self.qs).mean(-2)


class AvgLowPass(nn.Module):
    """ Average low passes at a given scattering order. """
    def __init__(self, r: int, sc_idxer: ScaleIndexer):
        super(AvgLowPass, self).__init__()
        self.mask_low = sc_idxer.low_pass_mask[r-1]

    @staticmethod
    def create_output_description(N:int, A: int, sc_idces: np.ndarray, sc_idxer: ScaleIndexer) -> pd.DataFrame:
        """ Return the dataframe that describes the output of forward. """
        df_n = pd.DataFrame(np.arange(N), columns=['n'])

        js = [sc_idxer.idx_to_path(idx) for idx in sc_idces if sc_idxer.is_low_pass(idx)]

        sc = pd.DataFrame([sc_idxer.path_to_idx(path[:-1] if len(path) > 1 else path) for path in js], columns=['sc'])
        js = pd.DataFrame([sc_idxer.idx_to_path(idx, squeeze=False) for idx in sc.sc.values],
                          columns=[f'j{o+1}' for o in range(sc_idxer.r_max)])
        low = pd.DataFrame([idx == sc_idxer.JQ(1) for idx in sc.values[:, 0]], columns=['low'])
        df_sc_js = pd.concat([sc, js, low], axis=1)
        df_sc_js['a'] = 0
        df_sc_js['r'] = [max(1, sc_idxer.r(idx) - 1) for idx in sc.values[:, 0]]

        return df_product(df_n, df_sc_js).reindex(columns=['r', 'n', 'sc', 'j1', 'j2', 'a', 'low'])

    def forward(self, x: torch.tensor) -> torch.tensor:
        """ Computes E[x x^T].

        :param x: B x N x js x A x T tensor
        :return: B x N x K tensor
        """
        return x[:, :, self.mask_low, :, :].mean(-1).reshape(x.shape[0], x.shape[1], -1)


class Cov(nn.Module):
    """ Diagonal model along scales. """
    def __init__(self, rl: int, rr: int, sc_idxer: ScaleIndexer, nchunks: int):
        super(Cov, self).__init__()
        self.sc_idxer = sc_idxer
        self.nchunks = nchunks

        self.df_scale = self.create_scale_description(sc_idxer.sc_idces[rl-1], sc_idxer.sc_idces[rr-1], sc_idxer)

        self.idx_l, self.idx_r = self.df_scale[['scl', 'scr']].values.T
        if rl == 2:
            self.idx_l -= sc_idxer.JQ(1) + 1
        if rr == 2:
            self.idx_r -= sc_idxer.JQ(1) + 1

    @staticmethod
    def create_scale_description(scls: np.ndarray, scrs: np.ndarray, sc_idxer: ScaleIndexer) -> pd.DataFrame:
        """ Return the dataframe that describes the scale association in the output of forward. """
        info_l = []
        for (scl, scr) in product(scls, scrs):
            rl, rr = sc_idxer.r(scl), sc_idxer.r(scr)
            ql, qr = sc_idxer.Qs[rl-1], sc_idxer.Qs[rr-1]
            if rl > rr:
                continue

            pathl, pathr = sc_idxer.idx_to_path(scl), sc_idxer.idx_to_path(scr)
            jl, jr = sc_idxer.idx_to_path(scl, squeeze=False), sc_idxer.idx_to_path(scr, squeeze=False)

            if rl == rr == 2 and pathl < pathr:
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

            info_l.append(('ps' if rl * rr == 1 else 'phaseenv' if rl * rr == 2 else 'envelope',
                           2, rl, rr, scl, scr, *jl, *jr, 0, 0, low or scl == scr, low))

        out_columns = ['c_type', 'q', 'rl', 'rr', 'scl', 'scr'] + \
                      [f'jl{r}' for r in range(1, sc_idxer.r_max + 1)] + \
                      [f'jr{r}' for r in range(1, sc_idxer.r_max + 1)] + \
                      ['al', 'ar', 'real', 'low']
        df_scale = pd.DataFrame(info_l, columns=out_columns)

        # now do a diagonal or cartesian product along channels
        df_scale = (
            df_scale
            .drop('jl2', axis=1)
            .rename(columns={'jr2': 'j2'})
        )

        return df_scale

    @staticmethod
    def cov(xl: torch.tensor, xr: torch.tensor) -> torch.tensor:
        """ Computes Cov(xl, xr).

        :param xl: B x Nl x K x T tensor
        :param xr: B x Nr x K x T tensor
        :return: B x Nl x Nr x K tensor, with Nr = 1 if diagonal model along channels
        """
        assert xl.shape[1] == xr.shape[1]
        return (xl * xr).mean(-1).unsqueeze(-3)

    def forward(self, sxl: torch.tensor, sxr: Optional[torch.tensor] = None) -> torch.tensor:
        """ Extract diagonal covariances j2=j'2.

        :param sxl: B x Nl x jl x Al x T tensor
        :param sxr: B x Nr x jr x Ar x T tensor
        :return: B x Nl x Nr x K tensor, with Nr = 1 if diagonal model along channels
        """
        if sxr is None:
            sxr = sxl

        scl, scr = self.idx_l, self.idx_r

        y_l = [
            self.cov(sxl[:, :, scl[chunk], 0, :], sxr[:, :, scr[chunk], 0, :].conj())
            for chunk in np.array_split(np.arange(self.df_scale.shape[0]), self.nchunks)
        ]

        return torch.cat(y_l, -1)


class CovScaleInvariant(nn.Module):
    """ Reduced representation by making covariances invariant to scaling. """
    def __init__(self, sc_idxer: ScaleIndexer, df_scale_input: pd.DataFrame):
        super(CovScaleInvariant, self).__init__()
        self.df_scale_input = df_scale_input
        self.register_buffer('P', self._construct_invariant_projector(sc_idxer.JQ(1)))

    @staticmethod
    def create_scale_description(sc_idxer) -> pd.DataFrame:
        """ Return the dataframe that describes the output of forward. """
        J = sc_idxer.JQ(1)

        data = []

        # phase-envelope coefficients
        for a in range(1, J):
            data.append((2, 1, 2, a, pd.NA, 0, 0, False, False, 'phaseenv'))

        # scattering coefficients
        for (a, b) in product(range(J-1), range(-J+1, 0)):
            if a - b >= J:
                continue
            data.append((2, 2, 2, a, b, 0, 0, a == 0, False, 'envelope'))

        df_output = pd.DataFrame(data, columns=['q', 'rl', 'rr', 'a', 'b', 'al', 'ar', 'real', 'low', 'c_type'])

        return df_output

    def _construct_invariant_projector(self, J) -> torch.tensor:
        """ The projector P that takes a scattering covariance matrix C and computes PC the invariant projection. """
        df = Description(self.df_scale_input)

        P_l = []

        # phase-envelope coefficients
        for a in range(1, J):
            P_row = torch.zeros(self.df_scale_input.shape[0], dtype=torch.complex64)
            for j in range(a, J):
                mask = df.where(jl1=j, jr1=j-a, c_type='phaseenv')
                assert mask.sum() == 1
                P_row[mask] = 1.0
            P_l.append(P_row)

        # scattering coefficients
        for (a, b) in product(range(J-1), range(-J+1, 0)):
            if a - b >= J:
                continue

            P_row = torch.zeros(self.df_scale_input.shape[0], dtype=torch.complex64)
            for j in range(a, J+b):
                mask = df.where(jl1=j, jr1=j-a, j2=j-b)
                assert mask.sum() == 1
                P_row[mask] = 1.0
            P_l.append(P_row)

        P = torch.stack(P_l)

        # to get average along j instead of sum
        P /= P.sum(-1, keepdim=True)

        return P

    def forward(self, cov: torch.tensor) -> torch.tensor:
        """
        Keeps the scale invariant part of a Scattering Covariance. It is obtained by projection.

        :param cov: B x Nl x Nr x K tensor
        :return:
        """
        return self.P @ cov
