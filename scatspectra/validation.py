""" Implement some standard statistics to validate the model. """
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from scatspectra.data_source import PriceData
from scatspectra.layers.statistics import Estimator
from scatspectra.utils import (
    shifted_product, windows, lighten_color, HMCPricer, Smile, implied_vol
)


class ValidationStatistics:

    @NotImplementedError
    def compute(self, data: PriceData, cuda: bool):
        pass
    
    @NotImplementedError
    def plot(self, ax: Axes, color: str, *args, **kwargs):
        pass


class Histogram(ValidationStatistics):
    """ Histogram of log-returns. """

    def __init__(self, 
        nbins: int, 
        left: float | None = None, 
        right: float | None = None, 
        logscale: bool = False
    ):
        self.nbins = nbins
        self.left = left
        self.right = right
        self.logscale = logscale

    def compute(self, data: PriceData, cuda: bool = False):
        return data,
    
    def plot(self, data, ax=None, color=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(4,4))
        bins = self.nbins
        if self.left is not None and self.right is not None:
            bins = np.linspace(self.left, self.right, num=self.nbins)
        ax.hist(data.dlnx.ravel(), bins=bins, color=color, density=True, alpha=0.5)
        if self.logscale:
            ax.set_yscale('log')
        ax.grid(True)
        return ax


class HistogramLogVol(ValidationStatistics):
    """ Histogram of the log of the unsigned log-returns |dlnx|. """

    def __init__(self, 
        nbins: int, 
        left: float | None = None, 
        right: float | None = None, 
        eps: float | None = 1e-6, 
        logscale: bool = False
    ):
        self.nbins = nbins
        self.left = left
        self.right = right
        self.eps = eps
        self.logscale = logscale

    def compute(self, data: PriceData, cuda: bool=False):
        return data,
    
    def plot(self, data, ax=None, color=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(4,4))
        bins = self.nbins
        if self.left is not None and self.right is not None:
            bins = np.linspace(self.left, self.right, num=self.nbins)
        ax.hist(np.log10(np.abs(data.dlnx)+self.eps).ravel(), bins=bins, color=color, density=True, alpha=0.5)
        if self.logscale:
            ax.set_yscale('log')
        ax.grid(True)
        return ax
    

class StructureFunctions(ValidationStatistics):
    """ Estimate the structure functions of the log-returns defined as 
    S(q, l) = < |lnx(t) - lnx(t-ell)|^q >_t. """

    def __init__(self, max_lag: int, qs: np.ndarray, normalize: bool=False):
        self.lags = torch.arange(1, max_lag)
        self.qs = torch.tensor(qs)
        self.normalize = normalize

    def compute(self, data: PriceData, cuda: bool = False):
        lnx = torch.tensor(data.lnx)
        qs = self.qs
        if cuda:
            lnx = lnx.cuda()
            qs = self.qs.cuda()

        if self.normalize:
            norm = lnx.diff().std()
            lnx /= norm

        sf = torch.zeros(len(self.lags), len(self.qs), device=lnx.device)

        for i, lag in enumerate(self.lags):  # TODO: replace for loop by convolution
            dlnx = torch.abs(lnx[:,:,lag:] - lnx[:,:,:-lag])
            if self.normalize:
                dlnx /= np.sqrt(lag)
            dlnx = dlnx[...,None] ** qs[None,None,None,:]  # b n t q
            sf[i, :] = dlnx.mean((0,1,2))

        return sf.cpu(), 
    
    def plot(self, sf, ax=None, color=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(4,4))
        if color is None:
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        no_plot = len(ax.lines) == 0
        for i, q in enumerate(self.qs):
            color_here = lighten_color(color, amount=np.linspace(0.4,1.0,self.qs.shape[0])[i])
            ax.plot(self.lags, sf[:,i], label=f"q={q:.1f}", color=color_here)
            # ax.scatter(self.lags, sf[:,i], s=10, marker='+')
        if no_plot: 
            ax.legend()
            ax.grid(True)
            ax.set_yscale('log')
            ax.set_xscale('log')
        return ax


def leverage(
    y: np.ndarray, 
    p: float, 
    taumax: int, 
    beta: float | None, 
    ksize: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimates < y(t-tau) |y(t)|^p >_t

    :param y: ... x T array of log-returns
    :param p: the power
    :param taumax: the maximum lag
    :param beta: smoothing parameter
    :param ksize: the size of the smoothing kernel
    :return:
    """
    tau_pos, tau_neg, pos, neg = shifted_product(x=y, y=np.abs(y)**p, beta=beta, ksize=ksize)
    tau = np.concatenate([tau_neg[1:taumax][::-1], tau_pos[:taumax]], axis=-1)
    lev = np.concatenate([np.flip(neg[..., 1:taumax], axis=-1), pos[..., :taumax]], axis=-1)
    return tau, lev


class Leverage(ValidationStatistics):
    """ Estimates the leverage correlation < dlnx(t-tau) |dlnx(t)|^p >_t """

    def __init__(
        self, 
        max_lag: int, 
        p: float = 1.0, 
        beta: float | None = None, 
        ksize: int | None = None, 
        standardize: bool = False
    ):
        self.max_lag = max_lag
        self.p = p
        self.beta = beta
        self.ksize = ksize
        self.standardize = standardize

    def compute(self, data: PriceData, cuda: bool = False):
        dlnx = data.dlnx.copy()
        
        if self.standardize:
            dlnx -= dlnx.mean()
            dlnx /= dlnx.std()

        # leverage correlation
        tau, lev = leverage(dlnx, self.p, self.max_lag, self.beta, self.ksize)
        tau = tau[self.max_lag//2:]
        lev = lev[...,self.max_lag//2:]
        lev = lev.mean((0,1))
        return tau, lev
    
    def plot(self, t_lev, lev, ax=None, color=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(4,4))
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.plot(t_lev, lev,color=color)
        ax.grid(True)
        return ax


class AverageSmile(ValidationStatistics):
    """ Compute the average smile and estimates quantities such as its Skew, 
    curvature, skew-stickyness ratio. 
    The skew_T and curvature_T are defined by the expansion of the smile around 
    the moneyness, skew_T = S_T / \sqrt{T}: 
    \sigma(T, M) = \sigma_ATM (1 + S_T M + curvature_T M^2 + o(M^2)).
    The skew-stickyness ratio estimation follows Vargas, Dao, Bouchaud, 2015
    https://arxiv.org/abs/1311.4078.
    """
    
    def __init__(self, 
        Ts: np.ndarray | list[int], 
        Ms: np.ndarray | list[float], 
        r: float, 
        ave: Estimator | None = None
    ):
        if np.abs(r) > 1e-6:
            raise NotImplementedError("Interest rate different from zero is not implemented yet.")
        self.Ts = np.array(Ts)
        self.Ms = np.array(Ms)
        self.r = r
        self.ave = ave

    def compute_avg_smile(self, data: PriceData) -> Smile:
        x = data.x.reshape(-1, data.x.shape[-1])

        pricer = HMCPricer(M=10, K_bounds=[60, 140], ave=None, detrend=True)

        vol = np.empty((len(self.Ts), len(self.Ms)))

        for iT, T in enumerate(self.Ts):

            # get the overlapping windows on which to price the option
            xT = windows(x, w=T+1, s=1)  # overlapping windows of size T+1 
            xT = xT.reshape(-1, T+1)
            xT = 100.0 * xT / xT[:, :1]  # rescale the collected price windows to start at 100

            # get the at-the-money implied vol at all maturities
            price_atm = pricer.price(x=xT, strike=100.0)[0]
            vol_atm = implied_vol(price_atm, 100.0, T/252, 100.0, r=0.0)
            Ks = Smile.from_M_to_K(vol_atm, T, self.Ms)

            for iK, K in enumerate(Ks):

                # get all the implied vols
                price = pricer.price(x=xT, strike=K)[0]
                vol[iT,iK] = implied_vol(price, K, T/252, 100.0, r=0.0)
        
        vol[ (vol<=1e-6) | (vol>=10.0) ] = np.nan

        return Smile(vol, Ts=self.Ts, Ks=None, rMness=self.Ms)
    
    def compute_skew_curvature_ssr(self, data: PriceData, smile: Smile) -> tuple[np.ndarray,...]:
        """ Compute the skew of the smile. """

        skew = np.empty(len(self.Ts))
        curvature = np.empty(len(self.Ts))
        ssr = np.empty(len(self.Ts))

        sigma = (np.abs(data.dlnx)**2).mean() ** 0.5
        tau, L = leverage(data.dlnx, p=2.0, taumax=75, ksize=25, beta=None)
        L_pos = L[:,:,tau>0].mean((0,1))

        for iT, T in enumerate(self.Ts):

            # polynomial fit of the smile 
            sli = slice(1,-1)
            poly_fit = np.poly1d(np.polyfit(smile.rMness[iT][sli], smile.vol[iT][sli], deg=5))
            vol_atm = poly_fit(0.0)

            # skew
            ST = poly_fit.deriv()(0.0) / vol_atm
            skew[iT] = ST / np.sqrt(T/252)

            # curvature 
            curvature[iT] = poly_fit.deriv().deriv()(0.0) / 2 / vol_atm

            # skew-stickyness ratio
            gT = L_pos[1:T+1].sum() / (2 * T * sigma ** 3)
            ssr[iT] = gT * np.sqrt(T) / ST

        return skew, curvature, ssr
    
    def compute(self, data, cuda=False):
        """ Compute the average smile.
        :param x: the long price time-series
        """
        # the average smile 
        avg_smile = self.compute_avg_smile(data)

        # the skew, curvature and skew-stickyness ratio
        skew, curvature, ssr = self.compute_skew_curvature_ssr(data, avg_smile)

        return avg_smile, skew, curvature, ssr

    def plot(self, avg_smile, skew, curvature, ssr, rescale=True, log=False, ax=None, color=None):
        """ Plot AVG smile, skew (absolute value), curvature and ssr. """
        if ax is None:
            _, ax = plt.subplots(1, 4, figsize=(8,2))
        color = color or 'blue'
        # plot avg smile
        avg_smile.plot(ax[0], color=color, rescale=rescale)
        # plot skewness
        ax[1].plot(self.Ts, np.abs(skew), color=color)
        # plot curvature
        ax[2].plot(self.Ts, curvature, color=color)
        # plot ssr
        ax[3].plot(self.Ts, ssr, color=color)
        for _ in ax:
            _.grid(True)
        if log:
            for _ in ax[1:]:
                _.set_xscale('log')
                _.set_yscale('log')
        return ax