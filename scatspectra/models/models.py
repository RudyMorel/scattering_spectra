from typing import List, Dict
from abc import abstractmethod
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from scatspectra.description import (
    ScaleIndexer, DescribedTensor, scattering_network_description,
    scattering_coefficients_description, build_description_mean_spars,
    build_description_correlation, make_description_compatible,
    create_scale_invariant_description
)
from scatspectra.layers import (
    Wavelet, Modulus, PhaseOperator, LinearLayer, NormalizationLayer,
    ScatteringCoefficients, AverageLowPass, Correlation,
    CorrelationScaleInvariant, Estimator
)
from scatspectra.utils import format_args_to_list, df_product_channel_single


ADMISSIBLE_MODEL_TYPES = [
    None,
    'scat_marginal',
    'scat_spectra',
    'inv_scat_spectra',
    'scat_marginal+scat_spectra'
]


class ChunkedModule(nn.Module):
    """ Manage chunks on batch dimension. """

    def __init__(self,
                 #  module: nn.Module,
                 nchunks: int) -> None:
        super(ChunkedModule, self).__init__()
        self.nchunks = nchunks

    @abstractmethod
    def forward_batch(self, x: torch.Tensor) -> DescribedTensor:
        """ Forward on the batch dimension. """
        pass

    def forward(self, x: torch.Tensor) -> DescribedTensor:
        """
        Chunked forward on the batch dimension.

        :param x: B x ... tensor
        :return:
        """
        nchunks = min(x.shape[0], self.nchunks)
        batch_split = np.array_split(np.arange(x.shape[0]), nchunks)
        Rxs = [self.forward_batch(x[bs, ...]) for bs in batch_split]
        return DescribedTensor(
            x=x,
            y=torch.cat([Rx.y for Rx in Rxs]),
            df=Rxs[0].df,
            config=Rxs[0].config
        )


class Model(ChunkedModule):
    """ Model class for analysis and generation. """

    def __init__(self,
                 model_type: str | None,
                 T: int,
                 r: int,
                 J: int | List[int],
                 Q: int | List[int],
                 wav_type: str | List[str],
                 wav_norm: str | List[str],
                 high_freq: float | List[float],
                 A: int | None,
                 rpad: bool,
                 channel_transforms: List[torch.Tensor] | None,
                 N: int,
                 Ns: List[int] | None,
                 diago_n: bool,
                 cross_params: Dict | None,
                 sigma2: torch.Tensor | None,
                 norm_on_the_fly: bool,
                 estim_operator: Estimator | None,
                 qs: List[float] | None,
                 coeff_types: List[str] | None,
                 dtype: torch.dtype | None,
                 histogram_moments: bool,
                 skew_redundance: bool,
                 nchunks: int,
                 **kwargs):
        super(Model, self).__init__(nchunks)

        self.config = locals().copy()

        self.model_type = model_type

        J, Q, wav_type, wav_norm, high_freq = \
            format_args_to_list(J, Q, wav_type, wav_norm, high_freq, n=r)

        self.sc_idxer = ScaleIndexer(
            r=r, J=J, Q=Q, strictly_increasing=not skew_redundance
        )
        self.r = r

        # time layers
        self.Ws = nn.ModuleList([
            Wavelet(T, J[o], Q[o], wav_type[o], wav_norm[o],
                    high_freq[o], rpad, o+1, self.sc_idxer)
            for o in range(r)
        ])

        # phase transform
        self.A = A
        self.phase_operator = Modulus() if A is None else PhaseOperator(A)

        # normalization layer
        if norm_on_the_fly:
            self.norm_layer = NormalizationLayer(2, None, True)
        elif sigma2 is not None:
            self.norm_layer = NormalizationLayer(2, sigma2.pow(0.5), False)
        else:
            self.norm_layer = nn.Identity()

        # channel transforms
        if channel_transforms is None:
            self.L1 = nn.Identity()
            self.L2 = nn.Identity()
            self.Lphix = nn.Identity()
        else:
            B1, B2, B3 = channel_transforms
            self.L1 = LinearLayer(B1)
            self.L2 = LinearLayer(B2)
            self.Lphix = LinearLayer(B3)
        self.N = N
        self.Ns = Ns or [N] * r
        self.diago_n = diago_n
        self.cross_params = cross_params

        # marginal moments i.e. computed on univariate scale channels
        self.histogram_moments = histogram_moments
        self.module_mean_spars = AverageLowPass(ave=estim_operator)
        self.module_scat = ScatteringCoefficients(
            qs=qs or [1.0],
            ave=estim_operator
        )
        self.module_scat_q1 = ScatteringCoefficients(
            qs=[1.0], ave=estim_operator)

        if r == 2:
            self.module_corr_w = Correlation(
                rl=1, rr=1,
                sc_idxer=self.sc_idxer,
                ave=estim_operator
            )
            self.module_corr_wmw = Correlation(
                rl=1, rr=2,
                sc_idxer=self.sc_idxer,
                ave=estim_operator
            )
            self.module_corr_mw = Correlation(
                rl=2, rr=2,
                sc_idxer=self.sc_idxer,
                ave=estim_operator
            )
            self.df_corr = build_description_correlation(
                [1, 1], self.sc_idxer, diago_n=True
            )
            self.module_corrinv = None
            if model_type == 'inv_scat_spectra':
                self.module_corrinv = CorrelationScaleInvariant(
                    sc_idxer=self.sc_idxer,
                    df_scale_input=self.df_corr,
                    skew_redundance=skew_redundance
                )

        self.df = self.build_description()

        self.all_coeff_types = None 
        self.coeff_types = None
        if 'coeff_type' in self.df.columns:
            self.all_coeff_types = self.df.coeff_type.unique().tolist()
            self.coeff_types = coeff_types or self.df.coeff_type.unique().tolist()

        # cast model to the right precision
        if dtype == torch.float64:
            self.double()

    def double(self):
        """ Change model parameters and buffers to double precision (float64 and complex128). """
        def cast(t):
            if t.is_floating_point():
                return t.double()
            if t.is_complex():
                return t.to(torch.complex128)
            return t
        return self._apply(cast)

    def compute_scattering(self, x: torch.Tensor) -> List[torch.Tensor]:
        """ Compute the Wx, W|Wx|, ..., W|...|Wx|| 
        i.e. standard scattering coefficients. """
        Sx_l = []
        for o, W in enumerate(self.Ws):
            x = W(x)
            if o == 0:
                x = self.norm_layer(x)
            Sx_l.append(x)
            x = torch.abs(x)

        return Sx_l

    def compute_mean_and_spars(self, Wx, reshape=True):
        """ Compute E{|Wx|}. """
        exp = self.module_mean_spars(Wx)
        if reshape:
            return exp.view(exp.shape[0], -1, exp.shape[-1])
        return exp

    def compute_phase_mod_correlation(self, Wx, WmWx, diago_n, reshape=True):
        """ Compute reduced phase-modulus correlation matrices
        E{Wx Wx^T}, E{Wx W|Wx|^T}, E{W|Wx| W|Wx|^T}"""
        corr1 = self.module_corr_w(Wx, Wx, diago_n=diago_n)
        corr2 = self.module_corr_wmw(Wx, WmWx, diago_n=diago_n)
        corr3 = self.module_corr_mw(WmWx, WmWx, diago_n=diago_n)

        def reshaper(y):
            if reshape:
                return y.view(y.shape[0], -1, y.shape[-1])
            return y

        return torch.cat([
            reshaper(corr) for corr in [corr1, corr2, corr3]
        ], dim=-2)

    def count_coefficients(self, query: str | None = None) -> int:
        """ Returns the number of moments satisfying kwargs. """
        df = self.df
        if self.coeff_types is not None:
            df = self.df[np.isin(self.df.coeff_type, self.coeff_types)]
        if query is not None:
            df = df.query(query)
        return df.shape[0]

    def build_description(self) -> pd.DataFrame:

        if self.model_type is None:

            df = pd.concat([
                scattering_network_description(o+1, self.Ns[o], self.sc_idxer)
                for o in range(self.r)
            ], axis=0)

        elif self.model_type == 'scat_marginal':

            qs = self.module_scat.qs.cpu().numpy()
            df = scattering_coefficients_description(self.Ns,
                                                     self.sc_idxer,
                                                     qs)

        elif self.model_type == 'scat_spectra':

            df_r1 = build_description_mean_spars(self.N, self.sc_idxer)
            df_r2 = build_description_correlation(
                self.Ns, self.sc_idxer, diago_n=self.diago_n
            )

            df = pd.concat([df_r1, df_r2])
            df.coeff_type = pd.Categorical(
                df.coeff_type,
                categories=['mean', 'spars', 'variance',
                            'skewness', 'kurtosis']
            )

        elif self.model_type == 'inv_scat_spectra':

            df_r1 = build_description_mean_spars(self.N, self.sc_idxer)
            df_r2 = build_description_correlation(
                [self.N, self.N], self.sc_idxer, diago_n=self.diago_n
            )

            # description of the scale NON-invariant part
            # 'mean', 'spars, ''variance' coefficients and all low-pass coefficients
            df_non_I = df_r2.query("is_low == True | coeff_type == 'variance'")
            df_non_I = pd.concat([df_r1, df_non_I])

            # phaseenv and envelope that are invariant
            df_inv = create_scale_invariant_description(self.sc_idxer)
            df_inv = df_product_channel_single(df_inv, self.N, method="same")

            # make non-invariant / invariant descriptions compatible
            df_inv['scr'] = df_inv['scl'] = pd.NA
            df_inv['jl1'] = df_inv['jr1'] = df_inv['j2'] = pd.NA
            df_non_I['a'] = df_non_I['b'] = pd.NA

            df = pd.concat([df_non_I, df_inv])

        elif self.model_type == 'scat_marginal+scat_spectra':

            df_exp = build_description_mean_spars(self.N, self.sc_idxer)

            df_scat = scattering_network_description(
                r=2, N_out=self.Ns[1], scale_indexer=self.sc_idxer
            )

            df_scat = make_description_compatible(df_scat)
            df_scat['q'] = 1
            df_scat['coeff_type'] = "scat_marginal"

            df_corr = build_description_correlation(
                self.Ns, self.sc_idxer, diago_n=self.diago_n
            )

            df = pd.concat([df_exp, df_scat, df_corr])
            df.coeff_type = pd.Categorical(
                df.coeff_type,
                categories=['mean', 'spars', 'variance', "scat_marginal",
                            'skewness', 'kurtosis']
            )

        else:

            raise ValueError(f"Unrecognized model type {self.model_type}.")

        return df

    def forward_batch(self, x: torch.Tensor):
        """
        :param x: tensor of shape (B, N, T)
        """

        if x.ndim != 3:
            raise ValueError("Wrong input tensor dimensions.")

        # add scale and phase dimensions
        x = x[:, :, None, None, :]

        # compute scattering coefficients Sx(t, j1 ... jr) for r=1,2,...
        Sx = self.compute_scattering(x)

        # compute scattering coefficients Sx(t, j1 ... jr)
        if self.model_type is None:

            y = torch.cat([
                y.view(x.shape[0], -1, x.shape[-1]) for y in Sx
            ], dim=1)

        # compute marginal scattering statistics <|Sx(t, j1 ... jr)|^q>_t
        elif self.model_type == 'scat_marginal':

            Sx = torch.cat([
                y.view(x.shape[0], -1, x.shape[-1]) for y in Sx
            ], dim=1)
            y = self.module_scat(Sx)
            y = y.view(y.shape[0], -1, y.shape[-1])

        # compute scattering spectra
        # <Sx(t, j1 ... jr)>_t, <Sx(t, j1 ... jr) Sx(t, j'1 ... j'r')^T>_t
        elif self.model_type == 'scat_spectra':

            # <x>_t and <|Wx(t,j)|>_t
            exp = self.compute_mean_and_spars(Sx[0])

            # <Sx(t, j1 ... jr) Sx(t, j'1 ... j'r')^T>_t
            corr = self.compute_phase_mod_correlation(
                *Sx, diago_n=self.diago_n
            )

            y = torch.cat([exp, corr], dim=1)

        # compute invariant scattering spectra
        # they are scale invariant (up to scale axis edge effects)
        # <Sx(t, j1 ... jr) Sx(t, j'1 ... j'r')^T>_{t,j1}
        elif self.model_type == 'inv_scat_spectra':

            exp = self.compute_mean_and_spars(Sx[0])

            noninv_mask = self.df_corr.eval(
                "coeff_type == 'variance' | is_low == True"
            ).values
            corr_full = self.compute_phase_mod_correlation(*Sx,
                                                           diago_n=self.diago_n,
                                                           reshape=False)
            corr_noninv = corr_full[..., noninv_mask, :]
            corr_inv = self.module_corrinv(corr_full)  # invariant to dilation

            corr = torch.cat([c.view(c.shape[0], -1, c.shape[-1])
                              for c in [corr_noninv, corr_inv]], dim=-2)

            y = torch.cat([exp, corr], dim=-2)

        # compute scattering marginal statistics and scattering spectra
        elif self.model_type == 'scat_marginal+scat_spectra':

            Wx, WmWx = Sx[:2]

            exp1 = self.compute_mean_and_spars(Wx, reshape=True)
            exp2 = self.module_scat_q1(WmWx)
            exp = torch.cat([
                exp1, exp2.view(exp2.shape[0], -1, exp2.shape[-1])
            ], dim=1)

            corr = self.compute_phase_mod_correlation(
                Wx, WmWx, diago_n=self.diago_n
            )

            y = torch.cat([exp, corr], 1)

        else:

            raise ValueError(f"Unrecognized model type {self.model_type}")

        if not y.is_complex():
            y = torch.complex(y, torch.zeros_like(y))

        if self.histogram_moments:
            # get multi-scale increments (can be seen as a particular wavelet transform)
            filters = torch.tensor([[1] * (2 ** j) + [0] * (x.shape[-1] - 2 ** j) for j in range(self.sc_idxer.J[0])],
                                   dtype=x.dtype, device=x.device)

            def multiscale_dx(x):
                return torch.fft.ifft(torch.fft.fft(filters[:, None, :]) * torch.fft.fft(x)).real

            dx = multiscale_dx(x)
            dx_norm = dx.pow(2.0).mean(-1, keepdim=True).pow(0.5)
            # def energy(x):
            #     return x.pow(2.0).mean((0,1,3,4)) / target_norm.pow(2.0).mean((0,1,3,4))

            def skewness(dx):
                # x_norm = x / target_norm
                # x2_signed = nn.ReLU()(x_norm).pow(2.0) - nn.ReLU()(-x_norm).pow(2.0)
                return torch.sigmoid(dx / dx_norm).mean(-1, keepdim=True)
            # def kurtosis(x):
            #     return torch.abs(x).mean((0,1,3,4)).pow(2.0) / target_norm.pow(2.0).mean((0,1,3,4))
            # histogram_statistics = [skewness]

            # histo_stats = torch.stack([stat(dxw_target) for stat in histogram_statistics])
            histo_stats = skewness(dx)
            histo_stats = np.sqrt(0.5) * histo_stats[:, 0, :, 0, :]

            y = torch.cat([y, histo_stats], dim=1)

        Rx = DescribedTensor(x=x, y=y, df=self.df)

        if self.coeff_types is not None:
            return Rx.query(coeff_type=self.coeff_types)

        return Rx
