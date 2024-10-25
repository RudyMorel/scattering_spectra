""" Moments to be used on top of a scattering transform. """
from typing import List, Iterable
from itertools import product
import pandas as pd
import torch
import torch.nn as nn

from scatspectra.description import create_scale_description, ScaleIndexer


class Estimator(nn.Module):
    """ Estimator used on scattering. """

    def forward(self, x: torch.Tensor, **kwargs):
        pass


class TimeAverage(Estimator):
    """ Averaging operator to estimate probabilistic expectations or correlations. """
    
    def __init__(self, window: Iterable | None = None) -> None:
        super(TimeAverage, self).__init__()
        self.w = None
        if window is not None:
            self.w = torch.tensor(window, dtype=torch.int64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.w is not None:
            x = x[..., self.w]
        return x.mean(-1, keepdim=True)


class WindowSelector(Estimator):
    """ Selecting operator. """

    def __init__(self, window: Iterable) -> None:
        super(WindowSelector, self).__init__()
        self.w = torch.tensor(window, dtype=torch.int64)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x[..., self.w]
    

class AvgPooling(Estimator):
    """ Average pooling operator for complex dtype tensor. """

    def __init__(self, kernel_size, stride):
        super(AvgPooling, self).__init__()
        self.pool = torch.nn.AvgPool1d(kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Average pooling operator for complex dtypes.

        :param x: any dimension tensor.
        :return:
        """
        if x.is_complex():

            # Separate real and imaginary parts.
            x_real = x.real
            x_imag = x.imag

            # Perform average pooling on real and imaginary parts separately.
            x_real = self.pool(x_real.view(x.shape[0], -1, x.shape[-1]))
            x_imag = self.pool(x_imag.view(x.shape[0], -1, x.shape[-1]))

            # Combine real and imaginary parts back into a complex tensor.
            y = torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1))

        else:

            y = self.pool(x.view(x.shape[0], -1, x.shape[-1]))

        return y.view(x.shape[:-1] + (-1, ))


class AverageLowPass(nn.Module):
    """ Compute average on scattering layers. """

    def __init__(self, ave: Estimator | None = None):
        super(AverageLowPass, self).__init__()
        self.ave = ave or TimeAverage()

    def forward(self, Wx: torch.Tensor) -> torch.Tensor:
        """ Computes <Wx>_t = <x>_t and <W|Wx|>_t = <|Wx|>_t.

        :param Wx: B x N x js x A x T tensor
        :return: B x N x K x T' tensor
        """
        y_mod = self.ave(torch.abs(Wx[:, :, :-1, :, :]))
        y_low = self.ave(Wx[:, :, -1:, :, :])

        y = torch.cat([y_mod, y_low], dim=-3)

        return y.reshape(y.shape[0], y.shape[1], -1, y.shape[-1])


class ScatteringCoefficients(nn.Module):
    """ Compute per channel (marginal) order q moments. """

    def __init__(self, qs: List[float], ave: Estimator | None):
        super(ScatteringCoefficients, self).__init__()
        self.coeff_types = ['marginal']
        self.ave = ave or TimeAverage()

        self.register_buffer('qs', torch.tensor(qs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes <|Sx|^q>_t.

        :param x: B x N x js x A x T tensor
        :return: B x N x js x A x len(qs) x T' tensor
        """
        return self.ave(torch.abs(x).unsqueeze(-2) ** self.qs[:, None])  # type: ignore


class Correlation(nn.Module):
    """ Diagonal model along scales. """

    def __init__(self,
        rl: int,
        rr: int,
        sc_idxer: ScaleIndexer,
        ave: Estimator | None = None
    ):
        super(Correlation, self).__init__()
        self.sc_idxer = sc_idxer

        self.df_scale = create_scale_description(
            sc_idxer.sc_idces[rl-1], sc_idxer.sc_idces[rr-1], sc_idxer)

        self.idx_l, self.idx_r = self.df_scale[['scl', 'scr']].values.T
        if rl == 2:
            self.idx_l -= sc_idxer.JQ(1) + 1
        if rr == 2:
            self.idx_r -= sc_idxer.JQ(1) + 1

        self.ave = ave or TimeAverage()

    @staticmethod
    def get_channel_idx(Nl: int, Nr: int, multivariate: bool):
        if multivariate:
            nl, nr = torch.tensor(list(product(range(Nl), range(Nr)))).T
        else:
            nl = nr = torch.arange(Nl)
        return nl, nr

    def forward(self, 
        sxl: torch.Tensor,
        sxr: torch.Tensor | None = None,
        multivariate: bool = True
    ) -> torch.Tensor:
        """ Extract diagonal covariances j2=j'2.

        :param sxl: B x Nl x jl x Al x T tensor
        :param sxr: B x Nr x jr x Ar x T tensor
        :return: B x channels x K x T' tensor
        """
        if sxr is None:
            sxr = sxl

        # select communicating scales
        scl, scr = self.idx_l, self.idx_r
        xl, xr = sxl[:, :, scl, 0, :], sxr[:, :, scr, 0, :]

        # select communicating channels
        nl, nr = self.get_channel_idx(sxl.shape[1], sxr.shape[1], multivariate)
        xl, xr = xl[:, nl, ...], xr[:, nr, ...]

        y = self.ave(xl * xr.conj())

        return y


class CorrelationScaleInvariant(nn.Module):
    """ Reduced representation by making correlations invariant to scaling. """

    def __init__(self, 
        sc_idxer: ScaleIndexer,
        df_scale_input: pd.DataFrame,
        skew_redundance: bool
    ):
        super(CorrelationScaleInvariant, self).__init__()
        self.df_scale_input = df_scale_input
        self.skew_redundance = skew_redundance
        self.register_buffer(
            'P', self._construct_invariant_projector(sc_idxer.JQ(1))
        )

    def _construct_invariant_projector(self, J: int) -> torch.Tensor:
        """ The projector P that takes a scattering covariance matrix C and computes PC the invariant projection. """

        df = self.df_scale_input

        P_l = []

        # skewness coefficients <Wx(t,j) W|Wx|(t,j-a,j)^*>_t
        for a in range(1-int(self.skew_redundance), J):
            P_row = torch.zeros(df.shape[0], dtype=torch.complex64)
            for j in range(a, J):
                mask = df.eval(
                    f"coeff_type=='skewness' & jl1=={j} & jr1=={j-a}"
                ).values
                assert mask.sum() == 1
                P_row[mask] = 1.0
            P_l.append(P_row)

        # kurtosis coefficients <W|Wx|(t,j,j-b) W|Wx|(t,j-a,j-b)^*>_t
        for (a, b) in product(range(J-1), range(-J+1, 0)):
            if a - b >= J:
                continue

            P_row = torch.zeros(df.shape[0], dtype=torch.complex64)
            for j in range(a, J+b):
                mask = df.eval(f"jl1=={j} & jr1=={j-a} & j2=={j-b}").values
                assert mask.sum() == 1
                P_row[mask] = 1.0
            P_l.append(P_row)

        P = torch.stack(P_l)

        # to get average along j instead of sum
        P /= P.sum(-1, keepdim=True)

        return P

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        """
        Keeps the scale invariant part of a Scattering Covariance. It is obtained by projection.

        :param cov: B x Nl x Nr x K tensor
        :return:
        """
        return self.P @ cov
