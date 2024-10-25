from abc import abstractmethod
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from scatspectra.description import (
    ScaleIndexer, DescribedTensor, scattering_network_description,
    scattering_coefficients_description, build_description_mean_spars,
    build_description_correlation, make_description_compatible,
    create_scale_invariant_description, build_description_histograms
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

    def __init__(self, nchunks: int) -> None:
        super(ChunkedModule, self).__init__()
        self.nchunks = nchunks

    @abstractmethod
    def forward_batch(
        self, x: torch.Tensor, bs: torch.Tensor | None = None
    ) -> DescribedTensor:
        """ Forward on a single batch. """
        pass

    def forward(self, x: torch.Tensor) -> DescribedTensor:
        """
        Chunked forward on the batch dimension.

        :param x: B x ... tensor
        :return:
        """
        nchunks = min(x.shape[0], self.nchunks)
        batch_split = torch.split(torch.arange(x.shape[0]), nchunks)
        Rxs = [self.forward_batch(x[bs, ...], bs) for bs in batch_split]
        return DescribedTensor(
            x=x,
            y=torch.cat([Rx.y for Rx in Rxs]),
            df=Rxs[0].df,
            config=Rxs[0].config
        )


class Model(ChunkedModule):
    """ Model class for analysis and generation. """

    def __init__(
        self,
        # generic model arguments
        model_type        : str | None,
        coeff_types       : list[str] | None,
        # time filtering arguments
        T                 : int,
        r                 : int,
        J                 : int | list[int],
        Q                 : int | list[int],
        wav_type          : str | list[str],
        wav_norm          : str | list[str],
        high_freq         : float | list[float],
        A                 : int | None,
        rpad              : bool,
        # multivariate arguments
        N                 : int,
        multivariate_model: bool,
        # statistics arguments
        estim_operator    : Estimator | None,
        qs                : list[float] | None,
        dtype             : torch.dtype | None,
        skew_redundance   : bool,
        histogram_moments : bool,
        # normalization arguments
        sigma2            : torch.Tensor | None,
        norm_on_the_fly   : bool,
        histogram_norm    : tuple[torch.Tensor] | None,
        # other arguments 
        nchunks           : int,
        gen_log_returns   : bool,
    ):
        """ 
        :param model_type: moments to compute on scattering
            None: compute Sx = (Wx, W|Wx|) and keep time axis t
            "scat_marginal": compute marginal statistics on Sx: <|Sx|^q>_t
            "scat_spectra": compute scattering spectra which corresponds to 
                <Sx>_t and <Sx Sx^T>_t
            "inv_scat_spectra": invariant scattering spectra, same as "scat_spectra" but now 
                consider the self-similar projection of matrix <Sx Sx^T>_t
            "scat_marginal+scat_spectra": both "scat_marginal" and "scat_spectra"
        :param coeff_types: coefficients to retain in the output
        :param T: length of the time-series
        :param r: number of convolutional layers in a scattering model
        :param J: number of scales (octaves) for each wavelet layer
        :param Q: number of wavelets per octave for each wavelet layer
        :param wav_type: wavelet type for each layer, e.g. 'battle_lemarie'
        :param wav_norm: wavelet normalization for each layer, e.g. 'l1'
        :param high_freq: central frequency of mother wavelet for each layer, 0.5 may lead to important aliasing
        :param A: number of angles for the phase transform, deprecated
        :param rpad: use a reflection pad to account for edge effects
        :param N: number of input channels (N>1: multivariate)
        :param multivariate_model: whether to consider correlations across input channels (i.e. across time-series)
        :param estim_operator: estimator to use for computing statistics, e.g. uniform time average
        :param qs: exponent to use in a "scat_marginal" or "scat+scat_spectra" model
        :param dtype: precision of the model
        :param skew_redundance: whether to choose jl1=jr1 for skewness coefficients
        :param histogram_moments: use histogram moments
        :param sigma2: normalization factor for the scattering coefficients
        :param norm_on_the_fly: normalize scattering coefficients on the fly
        :param histogram_norm: normalization factors for histogram moments
        :param nchunks: number of data chunks to process, increase it to reduce memory usage
        :param gen_log_returns: for histogram moments, need to know if the 
            model is applied on log-returns or log-prices
        """
        super(Model, self).__init__(nchunks)

        self.config = locals().copy()

        self.model_type = model_type
        self.gen_dlnx = gen_log_returns

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
        sigma = None if sigma2 is None else sigma2.pow(0.5)
        self.norm_layer = NormalizationLayer(2, sigma, norm_on_the_fly)
        self.hist_norm = histogram_norm
        if histogram_norm is not None:
            self.register_buffer('sigma_dlnx', histogram_norm[0].pow(0.5))
            self.register_buffer('sigma_lnmW', histogram_norm[1].pow(0.5))

        # multivariate
        self.N = N
        self.multivariate_model = multivariate_model

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
                1, self.sc_idxer, multivariate=False
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
        self.coeff_types = coeff_types
        if 'coeff_type' in self.df.columns:
            self.all_coeff_types = self.df.coeff_type.unique().tolist()

        # cast model to the right precision
        if dtype == torch.float32:
            self.float()
        elif dtype == torch.float64:
            self.double()
        else:
            raise ValueError(f"Dtype {dtype} not supported for Scattering model.")

    def double(self) -> 'Model':
        """ Change model parameters and buffers to double precision (float64 and complex128). """
        def cast(t):
            if t.is_floating_point():
                return t.double()
            if t.is_complex():
                return t.to(torch.complex128)
            return t
        return self._apply(cast)

    def compute_scattering_coefficients(
        self, x: torch.Tensor, bs: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """ Compute the Wx, W|Wx|, ..., W|...|Wx|| 
        i.e. standard scattering coefficients. """
        Sx_l = []
        for o, W in enumerate(self.Ws):
            x = W(x)
            if o == 0:
                x = self.norm_layer(x, bs)
            Sx_l.append(x)
            x = torch.abs(x)

        return Sx_l

    def compute_mean_and_spars(self, Wx: torch.Tensor, reshape: bool = True) -> torch.Tensor:
        """ Compute E{|Wx|}. """
        exp = self.module_mean_spars(Wx)
        if reshape:
            return exp.view(exp.shape[0], -1, exp.shape[-1])
        return exp

    def compute_phase_mod_correlation(self, 
        Wx: torch.Tensor, 
        WmWx: torch.Tensor, 
        multivariate: bool, 
        reshape: bool = True
    ) -> torch.Tensor:
        """ Compute reduced phase-modulus correlation matrices
        E{Wx Wx^T}, E{Wx W|Wx|^T}, E{W|Wx| W|Wx|^T}"""
        corr1 = self.module_corr_w(Wx, Wx, multivariate=multivariate)
        corr2 = self.module_corr_wmw(Wx, WmWx, multivariate=multivariate)
        corr3 = self.module_corr_mw(WmWx, WmWx, multivariate=multivariate)

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
                scattering_network_description(o+1, self.N, self.sc_idxer)
                for o in range(self.r)
            ], axis=0)

        elif self.model_type == 'scat_marginal':

            qs = self.module_scat.qs.cpu().numpy()
            df = scattering_coefficients_description(self.N, self.sc_idxer, qs)

        elif self.model_type == 'scat_spectra':

            df_r1 = build_description_mean_spars(self.N, self.sc_idxer)
            df_r2 = build_description_correlation(
                self.N, self.sc_idxer, multivariate=self.multivariate_model
            )

            df = pd.concat([df_r1, df_r2])
            df.coeff_type = pd.Categorical(
                df.coeff_type,
                categories=['mean', 'spars', 'variance',
                            'skewness', 'kurtosis']
            )

        elif self.model_type == 'inv_scat_spectra':

            df_r1 = build_description_mean_spars(self.N, self.sc_idxer)
            df_r2 = build_description_correlation(self.N, self.sc_idxer, multivariate=self.multivariate_model)

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
                r=2, N_out=self.N, scale_indexer=self.sc_idxer
            )

            df_scat = make_description_compatible(df_scat)
            df_scat['q'] = 1
            df_scat['coeff_type'] = "scat_marginal"

            df_corr = build_description_correlation(
                self.N, self.sc_idxer, multivariate=self.multivariate_model
            )

            df = pd.concat([df_exp, df_scat, df_corr])
            df.coeff_type = pd.Categorical(
                df.coeff_type,
                categories=['mean', 'spars', 'variance', "scat_marginal",
                            'skewness', 'kurtosis']
            )

        else:

            raise ValueError(f"Unrecognized model type {self.model_type}.")
        
        if self.histogram_moments:

            df_hist = build_description_histograms(self.sc_idxer)

            df = pd.concat([df, df_hist])

        return df

    def forward_batch(
        self, x: torch.Tensor, bs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        :param x: tensor of shape (B, N, T)
        :param bs: tensor of the indices b1,...,bn of the mini-batch
        """

        if x.ndim != 3:
            raise ValueError("Wrong input tensor dimensions.")

        # add scale and phase dimensions
        x = x[:, :, None, None, :]

        # compute scattering coefficients Sx(t, j1 ... jr) for r=1,2,...
        Sx = self.compute_scattering_coefficients(x, bs)

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
                *Sx, multivariate=self.multivariate_model
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
            corr_full = self.compute_phase_mod_correlation(
                *Sx, multivariate=self.multivariate_model, reshape=False
            )
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
                Wx, WmWx, multivariate=self.multivariate_model
            )

            y = torch.cat([exp, corr], 1)

        else:

            raise ValueError(f"Unrecognized model type {self.model_type}")

        if not y.is_complex():
            y = torch.complex(y, torch.zeros_like(y))

        if self.histogram_moments:
            # get multi-scale increments (can be seen as a particular wavelet transform)
            filters = torch.tensor(
                [[1] * (2 ** j) + [0] * (x.shape[-1]-1+self.gen_dlnx-2**j) for j in range(self.sc_idxer.J[0])], 
                dtype=x.dtype, device=x.device
            )

            def multiscale_dx(x):
                return torch.fft.ifft(torch.fft.fft(filters[:, None, :]) * torch.fft.fft(x)).real

            if self.gen_dlnx:
                dx = multiscale_dx(x)
            else:
                dx = multiscale_dx(x.diff(dim=-1))
            dx_target_norm = self.sigma_dlnx.permute(1, 0, 2)  # N J T

            # compute histogram statistics 
            skewness = torch.sigmoid(4*dx/dx_target_norm[None,:,:,None,:]).mean(-1, keepdim=True)[0,0,:,0,0]
            kurtosis = (dx/dx_target_norm[None,:,:,None,:]).abs().mean((0,1,3,4)).pow(2.0)
            log_envelopes = Sx[0].abs().add(1e-2).log()
            log_energy = log_envelopes.pow(2.0).mean((1, -2, -1))[...,None]
            log_energy = log_energy / self.sigma_lnmW.pow(2.0)[:,0,:,None]
            
            y = torch.cat([
                y,
                skewness[None,:,None],
                kurtosis[None,:,None],
                log_energy/10
            ], dim=1)

        Rx = DescribedTensor(x=x, y=y, df=self.df)

        if self.coeff_types is not None:
            return Rx.query(coeff_type=self.coeff_types)

        return Rx
