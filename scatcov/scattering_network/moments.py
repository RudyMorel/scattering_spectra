""" Moments to be used on top of a scattering transform. """
from typing import *
from itertools import product
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scatcov.scattering_network.scale_indexer import ScaleIndexer, ScatteringShape
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
    """ Average low passes. """
    def __init__(self, N: int, A: int, sc_idxer: ScaleIndexer):
        super(AvgLowPass, self).__init__()

        self.mask_low = np.array([sc_idxer.is_low_pass(idx) for idx in sc_idxer.get_all_idx()])

        self.df = self.get_output_description(N, A, sc_idxer)

    def get_output_description(self, N:int, A: int, sc_idxer: ScaleIndexer) -> pd.DataFrame:
        """ Return the dataframe that describes the output of forward. """
        df_n = pd.DataFrame(np.arange(N), columns=['n'])
        df_j = pd.DataFrame([sc_idxer.JQ(r=1)] + list(range(sc_idxer.JQ(r=1))), columns=['j'])
        df_a = pd.DataFrame(np.arange(A), columns=['a'])

        df = (
            df_n
            .merge(df_j, how='cross')
            .merge(df_a, how='cross')
        )
        return df

    def forward(self, x: torch.tensor) -> torch.tensor:
        """ Computes E[x x^T].

        :param x: B x N x js x A x T tensor
        :return: B x K x 1 tensor
        """
        avg = x[:, :, self.mask_low, :, :].mean(-1)

        return avg.view(x.shape[0], -1, 1)


class Cov(nn.Module):
    """ Diagonal model along scales. """
    def __init__(self, shape_l: ScatteringShape, shape_r: ScatteringShape,
                 sc_idxer: ScaleIndexer,
                 nchunks: int):
        super(Cov, self).__init__()
        self.shl = shape_l
        self.shr = shape_r
        self.sc_idxer = sc_idxer
        self.nchunks = nchunks
        assert shape_l.N == shape_r.N, "Diagonal covariance along channels requires same nb of channels."

        self.df_scale = self.get_output_description()

    def get_output_description(self) -> Description:
        """ Return the dataframe that describes the output of forward. """
        scat_idx_iter_l = product(self.sc_idxer.get_all_idx(), range(self.shl.A))
        scat_idx_iter_r = product(self.sc_idxer.get_all_idx(), range(self.shr.A))

        # first assemble a description for scales and phase indices
        info_l = []
        for ((scl, al), (scr, ar)) in product(scat_idx_iter_l, scat_idx_iter_r):
            rl, rr = self.sc_idxer.r(scl), self.sc_idxer.r(scr)
            if rl > rr:
                continue

            pathl, pathr = self.sc_idxer.idx_to_path(scl), self.sc_idxer.idx_to_path(scr)
            jl, jr = self.sc_idxer.idx_to_path(scl, squeeze=False), self.sc_idxer.idx_to_path(scr, squeeze=False)

            if jl < jr and rl == rr == 2:
                continue

            # correlate low pass with low pass or band pass with band pass and nothing else
            if (self.sc_idxer.is_low_pass(scl) and not self.sc_idxer.is_low_pass(scr)) or \
                    (not self.sc_idxer.is_low_pass(scl) and self.sc_idxer.is_low_pass(scr)):
                continue

            # only consider wavelets with non-negligibale overlapping support in Fourier
            # weak condition: last wavelets must be closer than one octave
            # if abs(path[-1] / ql - path_p[-1] / qr) >= 1:
            #     continue
            # strong condition: last wavelets must be equal
            if pathl[-1] != pathr[-1]:
                continue

            low = self.sc_idxer.is_low_pass(scl)

            info_l.append((2, rl, rr, scl, scr, *jl, *jr, al, ar, low or scl == scr, low,
                           'ps' if rl * rr == 1 else 'phaseenv' if rl * rr == 2 else 'envelope'))

        out_columns = ['q', 'rl', 'rr', 'scl', 'scr'] + \
                      [f'jl{r}' for r in range(1, self.sc_idxer.r_max + 1)] + \
                      [f'jr{r}' for r in range(1, self.sc_idxer.r_max + 1)] + \
                      ['al', 'ar', 're', 'low', 'c_type']
        df_scale = pd.DataFrame(info_l, columns=out_columns)

        # now do a diagonal or cartesian product along channels
        df_scale = (
            df_scale
            .drop('jl2', axis=1)
            .rename(columns={'jr2': 'j2'})
            .replace(-1, np.nan)
        )
        df_scale['j2'] = df_scale['j2'].astype('Int64')

        return Description(df_scale)

    def cov(self, xl: torch.tensor, xr: torch.tensor) -> torch.tensor:
        """ Computes Cov(xl, xr).

        :param xl: B x Nl x K x T tensor
        :param xr: B x Nr x K x T tensor
        :return: B x Nl x Nr x K tensor, with Nr = 1 if diagonal model along channels
        """
        return (xl * xr).mean(-1).unsqueeze(-3)

    def forward(self, sxl: torch.tensor, sxr: Optional[torch.tensor] = None) -> torch.tensor:
        """ Extract diagonal covariances j2=j'2.

        :param sxl: B x Nl x jl x Al x T tensor
        :param sxr: B x Nr x jr x Ar x T tensor
        :return: B x Nl x Nr x K x 1 tensor, with Nr = 1 if diagonal model along channels
        """
        if sxr is None:
            sxr = sxl

        scl, al, scr, ar = self.df_scale[['scl', 'al', 'scr', 'ar']].values.T

        y_l = [
            self.cov(sxl[:, :, scl[chunk], al[chunk], :], sxr[:, :, scr[chunk], ar[chunk], :].conj())
            for chunk in np.array_split(np.arange(self.df_scale.shape[0]), self.nchunks)
        ]
        y = torch.cat(y_l, -1)

        return y.unsqueeze(-1)


class CovScaleInvariant(nn.Module):
    """ Reduced representation by making covariances invariant to scaling. """
    def __init__(self, shape_l: ScatteringShape, shape_r: ScatteringShape, sc_idxer: ScaleIndexer, df_input: pd.DataFrame):
        super(CovScaleInvariant, self).__init__()
        self.shl = shape_l
        self.shr = shape_r
        self.sc_idxer = sc_idxer

        self.df_input = df_input
        self.df_output = self.get_output_description()

        self.register_buffer('P', self._construct_invariant_projector())

    def get_output_description(self) -> pd.DataFrame:
        """ Return the dataframe that describes the output of forward. """
        J = self.sc_idxer.JQ(1)

        data = []

        # phase-envelope coefficients
        for a in range(1, J):
            data.append((2, 2, 1, a, 1000000, 0, 0, False, False, 'phaseenv'))

        # scattering coefficients
        for (a, b) in product(range(J-1), range(-J+1, 0)):
            if a - b >= J:
                continue
            data.append((2, 2, 3, a, b, 0, 0, a == 0, False, 'envelope'))

        df_output = pd.DataFrame(data, columns=['q', 'rl', 'rr', 'a', 'b', 'al', 'ar', 're', 'low', 'c_type'])
        df_output = df_output.replace(1000000, np.nan)
        df_output['b'] = df_output['b'].astype('Int64')

        return df_output

    def _construct_invariant_projector(self) -> torch.tensor:
        """ The projector P that takes a scattering covariance matrix C and computes PC the invariant projection. """
        J = self.sc_idxer.JQ(1)
        df = Description(self.df_input)

        P_l = []

        # phase-envelope coefficients
        for a in range(1, J):
            P_row = torch.zeros(self.df_input.shape[0], dtype=torch.complex128)
            for j in range(a, J):
                mask = df.where(jl1=j, jr1=j-a, c_type='phaseenv')
                assert mask.sum() == 1
                P_row[mask] = 1.0
            P_l.append(P_row)

        # scattering coefficients
        for (a, b) in product(range(J-1), range(-J+1, 0)):
            if a - b >= J:
                continue

            P_row = torch.zeros(self.df_input.shape[0], dtype=torch.complex128)
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

        :param cov: B x Nl x Nr x K x 1 tensor
        :return:
        """
        return self.P @ cov
