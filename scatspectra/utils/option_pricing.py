from typing import Callable, Iterable
from pathlib import Path
import numpy as np
from scipy.stats import norm
from scipy.special import softmax
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from scatspectra.utils.plot_utils import lighten_color
from scatspectra.utils.array_utils import cumsum_zero


def price_BS(K, T, sigma, S0, r):
    """ Black-Scholes option pricing formula. """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / sigma / T ** 0.5
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / sigma / T ** 0.5

    price = (S0 * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))

    return price


def delta_BS(K, T, sigma, S0, r):
    """ Black-Scholes option delta hedge. """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / sigma / T ** 0.5

    return norm.cdf(d1, 0.0, 1.0)


class DiscreteProba:
    """ Discrete probability distribution. """
    def __init__(self, w: np.ndarray | None):
        self.w = w

    def avg(self, x: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
        return np.average(x, axis, self.w, keepdims=keepdims)

    def variance(self, x: np.ndarray, axis: int) -> np.ndarray:
        xmean = self.avg(x, axis, keepdims=True)
        return self.avg((x - xmean) ** 2, axis)

    def std(self, x: np.ndarray, axis: int) -> np.ndarray:
        return self.variance(x, axis) ** 0.5


class Softmax(DiscreteProba):
    def __init__(self, l2s: np.ndarray, eta: float):
        weights = softmax(-l2s ** 2 / 2 / eta ** 2)
        super(Softmax, self).__init__(weights)


class Uniform(DiscreteProba):
    def __init__(self):
        super(Uniform, self).__init__(None)

    def avg(self, x: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
        return x.mean(axis, keepdims=keepdims)


def implied_vol(
    price: float,
    K: float, 
    T: float, 
    S0: float, 
    r: float, 
    ignore_warning: bool = False
) -> float:
    """Given a price, compute the vol sigma in BS formula that matches this price."""
    a, b = 1e-6, 10.0
    func = lambda s: price_BS(K, T, s, S0, r) - price

    if func(a) > 0 or func(b) < 0:
        if not ignore_warning:
            print("WARNING. Implied_vol solver could not find a solution.")
        return a if func(a) > 0 else b
    return brentq(func, a, b)


class Smile:
    """ A smile is a collection of implied volatilities for different strikes 
    and maturities."""

    def __init__(self, 
        vol: list[np.ndarray], 
        Ts: np.ndarray | list, 
        Ks: list[np.ndarray] | None = None, 
        rMness: list[np.ndarray] | None = None, 
        date: str | None = None
    ):
        """ Ts: 1d array, vol: list of 1d arrays, Ks: list of 1d arrays, Ms: list of 1d arrays """
        self.date = date

        self.vol, self.Ts, Ks, rMness = self.format_input(vol, Ts, Ks, rMness)

        self.vol_atm = self.extract_vol_atm(self.vol, Ks, rMness)  # 1d array

        self.Ks, self.rMness = self.get_x_coordinates(self.vol_atm, self.Ts, Ks, rMness)

        self.prices = [price_BS(Ks, T / 252, sig, S0=100.0, r=0.0) for (Ks, T, sig) in zip(self.Ks, self.Ts, self.vol)]

    @staticmethod
    def format_input(vol, Ts, Ks, rMness):
        if Ks is None and rMness is None:
            raise ValueError("Smile object should have Ks or rMness specified.")
        Ts = np.array(Ts)

        def format(x):
            if isinstance(x, list):
                x = [np.array(arr, dtype=np.float64) for arr in x]
            if isinstance(x, np.ndarray) and x.dtype != np.float64:
                x = x.astype(np.float64)
            if isinstance(x, np.ndarray) and x.ndim == 2:
                x = list(x)
            if (isinstance(x, np.ndarray) and x.ndim == 1) or (
                    isinstance(x, list) and not isinstance(x[0], np.ndarray)):
                x = [np.array(x)] * Ts.size
            return x

        vol = format(vol)
        Ks = format(Ks)
        rMness = format(rMness)
        return vol, Ts, Ks, rMness

    def reduce_iTs(self, iTs: Iterable[int]) -> 'Smile':
        """ Return the smile that contains only the maturities selected through their index. """
        new_vol = [self.vol[iT] for iT in iTs]
        new_Ts = [self.Ts[iT] for iT in iTs]
        new_Ks = [self.Ks[iT] for iT in iTs]
        return Smile(vol=new_vol, Ts=new_Ts, Ks=new_Ks)

    @staticmethod
    def extract_vol_atm(
        vol: list[np.ndarray], 
        Ks: list[np.ndarray], 
        rMness: list[np.ndarray]
    ) -> np.ndarray:
        """ Extract the ATM volatilities from the volatilities by interpolating if necessary. """
        
        def get_atm_vol(volT, KsT, atm_defnition):
            if atm_defnition in KsT:
                i_atm = np.where(KsT == atm_defnition)[0][0]
                sigma_atm = volT[i_atm]
            else:
                ip = np.argmax(KsT > atm_defnition)
                lam = (atm_defnition - KsT[ip - 1]) / (KsT[ip] - KsT[ip - 1])
                sigma_atm = (1 - lam) * volT[ip] + lam * volT[ip - 1]
            return sigma_atm

        if Ks is not None:
            vol_atm = np.array([get_atm_vol(v, K, 100.0) for (v, K) in zip(vol, Ks)])
        else:
            vol_atm = np.array([get_atm_vol(v, M, 0.0) for (v, M) in zip(vol, rMness)])

        return vol_atm

    @staticmethod
    def from_K_to_M(
        vol_atm: float,
        T: int | float,
        K: np.ndarray | float
    ) -> np.ndarray:
        """ From strikes to rescaled log-moneyness. """
        sigma_sqrtT = vol_atm * (T / 252) ** 0.5
        M = np.log(K / 100.0) / sigma_sqrtT
        return M

    @staticmethod
    def from_M_to_K(
        vol_atm: float, 
        T: int | float, 
        rMness: np.ndarray | float
    ) -> np.ndarray:
        """ From rescaled log-moneyness to strikes. """
        sigma_sqrtT = vol_atm * (T / 252) ** 0.5
        K = 100.0 * np.exp(rMness * sigma_sqrtT)
        return K

    @staticmethod
    def get_x_coordinates(
        vol_atm: np.ndarray | float, 
        Ts: np.ndarray | list[float], 
        Ks: list[np.ndarray] | None, 
        rMness: list[np.ndarray] | None
    ) -> tuple[list, list]:
        """ Get the x coordinates (Ks or rMness) for the smile. """
        sigma_sqrtT = vol_atm * (Ts / 252) ** 0.5
        if Ks is None:
            Ks = [100.0 * np.exp(M * norm) for norm, M in zip(sigma_sqrtT, rMness)]
        if rMness is None:
            rMness = [np.log(K / 100.0) / norm for norm, K in zip(sigma_sqrtT, Ks)]
        return Ks, rMness

    def interpolate_Ks(self, iT: int, k: float) -> float:
        mask = np.isnan(self.vol[iT])
        return np.interp(k, self.Ks[iT][~mask], self.vol[iT][~mask])

    def interpolate_Ms(self, iT: int, m: float) -> float:
        mask = np.isnan(self.vol[iT])
        return np.interp(m, self.rMness[iT][~mask], self.vol[iT][~mask])

    def get(self, 
        T: int | float, 
        K: float | None = None, 
        rMness: float | None = None
    ) -> float:
        """ Obtain smile values by interpolation. """
        if not ((K is None) ^ (rMness is None)):
            raise ValueError("Should provide strike, log-moneyness or rescaled log-moneyness.")

        # maturity interpolation
        if T not in self.Ts:
            if T <= np.min(self.Ts) or T >= np.max(self.Ts):
                raise ValueError(f"Required maturity {T} is out of scope, min={self.Ts.min()}, max={self.Ts.max()}")
            iT2 = np.argmax(T < self.Ts)
            iT1 = iT2 - 1
            lam = (T - self.Ts[iT1]) / (self.Ts[iT2] - self.Ts[iT1])
        else:
            iT1 = iT2 = np.where(self.Ts == T)[0][0]
            lam = 1.0

        # strike interpolation
        interpolator = self.interpolate_Ks if K is not None else self.interpolate_Ms
        x = K if K is not None else rMness
        vol1 = interpolator(iT1, x)
        vol2 = interpolator(iT2, x)
        vol = (1 - lam) * vol1 + lam * vol2

        return vol

    def save(self, file_path: Path) -> None:
        if file_path.suffix != '.npz' or file_path.suffix == '':
            file_path = file_path.with_suffix('.npz')
        np.savez(file_path, Ts=self.Ts, Ks=np.array(self.Ks, dtype=object), vol=np.array(self.vol, dtype=object))

    @staticmethod
    def load(file_path: Path) -> 'Smile':
        ld = dict(np.load(file_path, allow_pickle=True))
        return Smile(**ld)

    def plot(self, 
        ax: Axes | None = None, 
        rescale: bool = False, 
        sub_idces: list[int] | None = None, 
        color: str = 'blue', 
        legend: bool = False
    ) -> Axes:

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 5))

        for i_T, T in enumerate(self.Ts):
            if sub_idces is not None and i_T not in sub_idces:
                continue

            color_here = lighten_color(color, amount=1 - np.linspace(0, 0.8, self.Ts.size)[i_T])
            xs = self.rMness[i_T] if rescale else self.Ks[i_T]
            ax.plot(xs, self.vol[i_T], color=color_here, label=f"T={T}")
            ax.scatter(xs, self.vol[i_T], color=color_here, marker="+")

        ax.grid(True)
        if legend:
            ax.legend()

        return ax


class HMCPricer:
    """ A class to price European options using Hedged Monte-Carlo (HMC),
    see Potters, Bouchaud, Sestovic, 2018.
    https://arxiv.org/abs/cond-mat/0008147v1 """

    def __init__(
        self,
        M: int,
        ave: DiscreteProba | None = None,
        detrend: bool | None = False,
        K_bounds: list | None = None,
        basis_func_method: str = 'piecewise_quadratic'
    ):
        self.ave = ave or Uniform()
        self.M = M
        self.detrend = detrend
        self.K_bounds = K_bounds
        self.basis_func_method = basis_func_method

    def reset_Ks(self, dt: int) -> None:
        if self.K_bounds is not None:
            Ka, Kb = self.K_bounds
            self.Ks = np.linspace(Ka, Kb, self.M) + 10
        else:
            M_max = 8.0
            Ms = np.linspace(-M_max, M_max, self.M)
            self.Ks = Smile.from_M_to_K(0.2, dt, Ms)

    def get_price(self, x: np.ndarray) -> np.ndarray:
        """ Get prices C(xs). """
        if self.basis_func_method != 'piecewise_quadratic':
            raise ValueError("Unkwnown basis function method in HMC")

        return np.stack([(x > k) * (x - k) ** 2 for k in self.Ks])

    def get_hedge(self, x: np.ndarray) -> np.ndarray:
        """ Get hedges phi(xs). """
        if self.basis_func_method != 'piecewise_quadratic':
            raise ValueError("Unkwnown basis function method in HMC")

        return np.stack([2 * (x > k) * (x - k) for k in self.Ks])

    def perform_iteration(
        self,
        it: int,
        x_curr: np.ndarray,
        x_prev: np.ndarray,
        discount: float,
        param_curr: np.ndarray | None,
        price_curr: np.ndarray | None
    ) -> np.ndarray:
        """ From param_curr get param_prev. They are obtained by linear regression of price_curr
        against previous price + previous hedge. """
        if param_curr is None:
            Y = price_curr  # C_T(x)
        elif price_curr is None:
            Y = param_curr @ self.get_price(x_curr)  # C_{k+1}(x_{k+1})
        else:
            raise ValueError("Step in HMC requires price_curr or param_curr")

        self.reset_Ks(it)

        # C_k + phi_k (x_{k+1} - x_k)
        X = discount * self.get_price(x_prev) + self.get_hedge(x_prev) * (discount * x_curr - x_prev)[None, :]

        regr = LinearRegression(fit_intercept=False)
        regr.fit(X.T, Y)
        param_prev = regr.coef_

        # param_test = scipy.linalg.pinv(X @ X.T) @ X @ Y

        return param_prev

    def price(
        self,
        x: np.ndarray,
        strike: float
    ) -> tuple[float, Callable, Callable]:
        """ Price a European Call option given several price paths X.

        :param x: array of shape R x (N+1) with R the number of price paths and N the number of days
        """
        N = x.shape[-1] - 1

        y = x.copy()
        
        # if the price paths have a consistent trend
        # substracting a deterministic path doesn't change the option price
        if self.detrend:
            dlny = np.diff(np.log(x), axis=-1)
            dlny -= dlny.mean(0, keepdims=True)
            lny = cumsum_zero(dlny)
            y = 100.0 * np.exp(lny)
        else:
            y = x

        # iterate backward to determine current params in Longstaff-Schwartz
        price_curr = (y[:, -1] - strike) * (y[:, -1] > strike)
        param_curr = None
        it = 1
        for n in np.arange(1, N + 1)[::-1]:
            #             discount = np.exp(-r * T / N)  # TODO: Verify the discount, T / N or N / T ?
            discount = 1.0
            param_curr = self.perform_iteration(it, y[:, n], y[:, n-1], discount, param_curr, price_curr if n == N else None)
            it += 1

        # go back to the price
        prices = param_curr @ self.get_price(y[:, 0])

        return self.ave.avg(prices, axis=-1), lambda x: param_curr @ self.get_price(x), lambda x: param_curr @ self.get_hedge(x)


def compute_smile(
    x: np.ndarray, 
    Ts: np.ndarray, 
    Ms: np.ndarray, 
    r: float = 0.0,
    ave: DiscreteProba | None = None
) -> Smile:
    """ Compute a smile from historical price data x through Hedged Monte-Carlo (HMC). 

    :param x: array of shape (R, T) with R the number of price paths and T number of days
    :param Ts: array of maturities (in days)
    :param Ms: array of log moneyness
    :param r: interest rate
    :param ave: the averaging operator to use in a HMC, uniform average if None
    :return: 
    """
    if np.abs(r) > 1e-6:
        raise NotImplementedError("Interest rate different from zero is not implemented yet.")

    vol_impli = np.zeros((len(Ts), len(Ms)))
    
    # pricer object
    pricer = HMCPricer(M=10, K_bounds=[60,140], ave=ave, detrend=True)
        
    for i_T, T in enumerate(Ts):

        xT = 100.0 * x[:,:T+1] / x[:,:1]

        # determine the ATM vol (needed for rescaled log-moneyness)
        price_atm = pricer.price(x=xT, strike=100.0)[0]
        vol_atm = implied_vol(price_atm, 100.0, T/252, 100.0, r)

        # adapt the strike in function 
        Ks_adapted = Smile.from_M_to_K(vol_atm, T, Ms)
        
        for i_K, K in enumerate(Ks_adapted):
            option_price = pricer.price(x=xT, strike=K)[0]
            vol_impli[i_T, i_K] = implied_vol(option_price, K, T/252, 100.0, r)
    
    # put failed optimization to NaN
    vol_impli[ (vol_impli<=1e-6) | (vol_impli>=10.0) ] = np.nan
        
    return Smile(vol=vol_impli, Ts=Ts, Ks=None, rMness=Ms)
