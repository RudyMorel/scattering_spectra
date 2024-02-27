""" Classes for handling time-series datasets. 
Notations
- B: batch size (i.e. number of realizations of a process)
- N: number of data channels (N>1 : multivariate process)
- T: number of data samples (i.e time samples)
e.g. a time-series dataset batch would typically be a (B,N,T) array. """

from typing import List, Tuple, Dict
from collections import OrderedDict
import shutil
from pathlib import Path
from tqdm import tqdm
import torch.multiprocessing as mp
from pathlib import Path
import math
import numpy as np
import pandas as pd

from scatspectra.utils import list_split
from scatspectra.standard_models import fbm, mrw, skewed_mrw, poisson_mu

from scatspectra.data import snp_data


class TimeSeriesDataset:
    """Time-series dataset stored in a directory. Each file in the directory
    should contain an array of same shape (B,N,T) with B the batch size, N the
    number of time-series channels and T the number of samples."""

    def __init__(
        self,
        dpath: Path,
        R: int,
        load: bool = False,
        slices: Dict[str, slice] | None = None,
        batch_shape=None,
    ):
        """
        :param dpath: path to the directory containing time-series
        :param R: number of time series to load
        :param load: load data at initialization
        :param slices: the slices to apply to each batch file
        """
        self.dpath = dpath

        if not any(dpath.iterdir()):
            raise ValueError("Dataset is empty.")

        if R is not None and slices is not None:
            assert R == sum(
                [sl.stop - sl.start for sl in slices.values()]
            ), "R and slices are not consistent."

        self.R = R

        # shape of a data batch
        self.B, self.N, self.T = batch_shape or self.infer_shape(dpath)

        # file paths that will be loaded
        all_fpaths = list(dpath.iterdir())
        if slices is None:
            self.fpaths = all_fpaths[: math.ceil(R / self.B)]
            self._init_slices()
        else:
            self.fpaths = [dpath / k for k in slices.keys()]
            self._slices = slices

        R_available = len(all_fpaths) * self.B
        if R > R_available:
            raise ValueError(
                f"The dataset contains only {R_available} time-"
                + f"series but {R} required."
            )

        # load data
        self.x = None
        if load:
            self.load()

    @staticmethod
    def infer_shape(dpath) -> Tuple[int, int, int]:
        """Infer the shape of a time-series batch in the directory."""
        return np.load(next(dpath.iterdir())).shape

    def _init_slices(self) -> None:
        """Default slices to apply to each batch."""
        self._slices = OrderedDict(
            {fp.name: slice(0, self.B) for fp in self.fpaths[:-1]}
        )
        last_stop = self.R % self.B + self.B * (self.R % self.B == 0)
        self._slices[self.fpaths[-1].name] = slice(0, last_stop)

    def load(self) -> np.ndarray:
        """Load the dataset."""

        # load only once
        if self.x is None:

            # load data
            x_l = [np.load(fp)[self._slices[fp.name], :, :] for fp in self.fpaths]

            # concatenate
            x = np.concatenate(x_l)
            self.x = x

        return self.x

    def split(self, num_splits: int) -> List["TimeSeriesDataset"]:
        """Split this object into almost equal objects."""
        fpaths_splits = list_split(self.fpaths, num_splits)
        slices_splits = [
            OrderedDict({fp.name: self._slices[fp.name] for fp in fps})
            for fps in fpaths_splits
        ]
        Rs = [
            sum([sl.stop - sl.start for sl in slices.values()])
            for slices in slices_splits
        ]
        batch_shape = self.B, self.N, self.T
        return [
            TimeSeriesDataset(self.dpath, R, False, slices, batch_shape)
            for (R, slices) in zip(Rs, slices_splits)
        ]


def cumsum_zero(dx):
    """Cumsum of a vector preserving dimension through zero-pading."""
    res = np.cumsum(dx, axis=-1)
    res = np.concatenate([np.zeros_like(res[..., 0:1]), res], axis=-1)
    return res


class PriceData:
    """Handle positive price time-series."""

    def __init__(
        self,
        x: np.ndarray | None = None,
        dx: np.ndarray | None = None,
        lnx: np.ndarray | None = None,
        dlnx: np.ndarray | None = None,
        x_init: float | np.ndarray | None = None,
        dts=None,
    ):
        """
        :param x: prices
        :param dx: price increments
        :param lnx: log prices
        :param dlnx: log price increments
        :param x_init: initial price values
        :param dts: time dates
        """
        if x is None:
            if dx is not None:
                x = self.from_dx_to_x(dx)
            elif lnx is not None:
                x = self.from_ln_to_x(lnx)
            elif dlnx is not None:
                x = self.from_dln_to_x(dlnx)
            else:
                raise ValueError(
                    "One and only one argument x,dx,lnx,dlnx" + " should be provided."
                )

        if x_init is not None:
            x_init = np.array(x_init)

        if (
            x_init is not None
            and isinstance(x_init, np.ndarray)
            and (x_init.ndim > 0)
            and x_init.shape != x[..., 0].shape
        ):
            raise ValueError("Wrong x_init format in PriceData.")

        self.dts = dts

        # set correct initial value through multiplication
        self.x = self.rescale(x, x_init, additive=dx is not None)
        self.lnx = np.log(x)
        self.dx = np.diff(x)
        self.dlnx = np.diff(np.log(x))

    @staticmethod
    def rescale(x: np.ndarray, x_init: np.ndarray | None, additive: bool) -> np.ndarray:
        """Impose the right starting point to each time-series in x."""
        if x_init is not None:
            if additive:
                x = x - x[..., :1] + x_init[..., None]
            else:
                x *= x_init[..., None] / x[..., :1]
        return x

    @staticmethod
    def from_dx_to_x(dx: np.ndarray):
        return cumsum_zero(dx)

    @staticmethod
    def from_ln_to_x(lnx: np.ndarray):
        return np.exp(lnx)

    @staticmethod
    def from_dln_to_x(dlnx: np.ndarray):
        lnx = cumsum_zero(dlnx)
        return PriceData.from_ln_to_x(lnx)


class DataGeneratorBase:
    """Base multi-processing dataset generator."""

    def __init__(
        self, model_name: str, B: int, cache_path: Path | None = None, **kwargs
    ):
        """
        :param model_name: name of the generative model
        :param B: size of generated batches
        :param cache_path: path to the cache directory if not set to "None"
        :param num_workers: number of workers for multi-processing
        """
        self.model_name = model_name
        self.B = B

        self.config = {"B": B, **kwargs}

        self.dpath = None
        if cache_path:
            self.dpath = cache_path / self.dirname(**kwargs)
            if len(str(self.dpath)) > 255:
                raise ValueError(f"Path is too long ({len(str(self.dpath))} > 250).")

        # Create cache directory.
        if self.dpath:
            self.dpath.mkdir(parents=True, exist_ok=True)

    def dirname(self, **kwargs) -> str:
        """Define directory name for this model with its params (kwargs)."""

        def format_path(key, value):
            if isinstance(value, dict):
                return "".join([format_path(k, v) for (k, v) in value.items()])
            elif value is None:
                return "_"
            elif isinstance(value, str):
                if value == "True":
                    value = 1
                elif value == "False":
                    value = 0
                return f"_{key[:2]}_{value}"
            elif isinstance(value, int):
                return f"_{key[:2]}{int(value)}"
            elif isinstance(value, float):
                return f"_{key[:2]}{value:.1e}"
            elif isinstance(value, bool):
                return f"_{key[:2]}{int(value)}"
            elif key == "coeff_types":  # TODO: make this more general
                return "".join(sorted("".join([v[:2] for v in value])))
            else:
                return ""

        dname = self.model_name + f"_B{self.B}" + format_path(None, kwargs)
        dname = dname.replace(".", "_").replace("-", "_").replace("+", "_")
        return dname

    def generate_batch(self, i: int) -> np.ndarray:
        raise NotImplemented

    def worker(self, i: int):
        """Generate a batch of time-series and save it if cache is used."""
        np.random.seed(None)
        try:
            x = self.generate_batch(i)
            if self.dpath is None or not self.dpath.is_dir():
                return (i, x)
            fname = f"{np.random.randint(int(1e5), int(1e6))}.npy"
            if (self.dpath / fname).is_file():  # very unlikely
                raise OSError(f"File {fname} already exists.")
            np.save(str(self.dpath / fname), x)
            return (i, None)
        except Exception as e:
            raise e

    def _generate(self, nbatches: int, num_workers: int):
        """Generate data in parallel and save it if cache is used.

        :param nbatches: number of data batches to generate
        :param num_workers: number of parallel workers
        """
        print(f"Model {self.model_name}: generating data ...")
        x_l = []
        try:
            mp.set_start_method("spawn")  # TODO: seems to slow down execution
        except RuntimeError:
            pass
        with mp.Pool(processes=num_workers) as pool:
            with tqdm(total=nbatches) as pbar:
                for result in pool.imap_unordered(self.worker, list(range(nbatches))):
                    _, x = result
                    x_l.append(x)
                    pbar.update()
        print("Finished.")

        return x_l

    def load(self, R: int, num_workers: int = 1) -> np.ndarray:
        """Load data, generate it if cache is used.

        :param R: number of realizations (number of time-series)
        :param num_workers: number of parallel workers
        """
        # if there is a cache, use it
        if self.dpath is not None and self.dpath.is_dir():
            print(
                f"Model {self.model_name}: "
                + f"using cache directory {self.dpath.name}."
            )
            # generate additional data if needed
            nbatches_avail = sum(1 for _ in self.dpath.iterdir())
            if nbatches_avail * self.B < R:
                nbatches = math.ceil(R / self.B) - nbatches_avail
                self._generate(nbatches, num_workers)

            return TimeSeriesDataset(self.dpath, R, True).load()

        # otherwise, generate data
        nbatches = math.ceil(R / self.B)
        x_l = self._generate(nbatches, num_workers)
        x = np.concatenate(x_l, axis=0)[:R, :, :]

        return x

    def erase_cache(self) -> None:
        """Erase cache if it exists."""
        if self.dpath is not None and self.dpath.is_dir():
            shutil.rmtree(self.dpath)


class PoissonGenerator(DataGeneratorBase):
    """Poisson jump process."""

    def __init__(
        self,
        T: int,
        mu: float,
        signed: bool = False,
        B: int = 64,
        cache_path: Path | None = None,
        **kwargs,
    ):
        """
        :param T: number of time samples
        :param mu: intensity of the Poisson process
        :param signed: if True then increments are +1/-1 uniformly
        :param B: batch size
        """
        super(PoissonGenerator, self).__init__(
            "poisson", B, cache_path, T=T, mu=mu, signed=signed, **kwargs
        )

    def generate_batch(self, i: int) -> np.ndarray:
        return poisson_mu(**self.config)[:, None, :]


class FBmGenerator(DataGeneratorBase):
    """Fractional Brownian motion."""

    def __init__(
        self, T: int, H: float, B: int = 64, cache_path: Path | None = None, **kwargs
    ):
        """
        :param T: number of time samples
        :param H: Hurst exponent
        :param B: batch size
        """
        super(FBmGenerator, self).__init__("fbm", B, cache_path, T=T, H=H, **kwargs)

    def generate_batch(self, i: int) -> np.ndarray:
        return fbm(**self.config)[:, None, :]


class MRWGenerator(DataGeneratorBase):
    """Multifractal random walk."""

    def __init__(
        self,
        T: int,
        H: float,
        lam: float,
        B: int = 64,
        cache_path: Path | None = None,
        **kwargs,
    ):
        """
        :param T: number of time samples
        :param H: Hurst exponent
        :param lam: intermittency parameter, if lam=0 then becomes a fBm
        :param B: batch size
        :param cache_path: _description_, defaults to None
        """
        super(MRWGenerator, self).__init__(
            "MRW", B, cache_path, T=T, L=T, H=H, lam=lam, **kwargs
        )

    def generate_batch(self, i: int) -> np.ndarray:
        return mrw(**self.config)[:, None, :]


class SMRWGenerator(DataGeneratorBase):
    """Skewed Multifractal Random Walk."""

    def __init__(
        self,
        T: int,
        H: float,
        lam: float,
        K0: float = 0.035,
        alpha: float = 0.23,
        beta: float = 0.5,
        gamma: float = 1 / (2**12) / 64,
        B: int = 64,
        cache_path: Path | None = None,
        **kwargs,
    ):
        """
        :param T: number of time samples
        :param H: Hurst exponent
        :param lam: intermittency parameter, if lam=0 then becomes a fBm
        :param B: batch size
        :param cache_path: _description_, defaults to None
        """
        super(SMRWGenerator, self).__init__(
            "SMRW",
            B,
            cache_path,
            T=T,
            L=T,
            H=H,
            lam=lam,
            K0=K0,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            **kwargs,
        )

    def generate_batch(self, i: int) -> np.ndarray:
        return skewed_mrw(**self.config)[:, None, :]


class SPDaily(PriceData):
    """S&P500 daily prices based on data obtained from the Wall Street Journal
    https://www.wsj.com/market-data/quotes/index/SPX/historical-prices"""

    def __init__(
        self, start: str = "03-01-2000", end: str = "07-02-2024", **kwargs
    ) -> None:
        """Initialize the dataset.

        Args:
            start (str, optional): start date. Defaults to '03-01-2000'.
            end (str, optional): end date. Defaults to '07-02-2024'.

        Raises:
            ValueError: if selected dates are out of range for available data.
        """
        # load full time-series
        df = snp_data

        max_date = df.index.max()
        min_date = df.index.min()

        # select dates
        start = pd.to_datetime(start, dayfirst=True)
        end = pd.to_datetime(end, dayfirst=True)

        if start < min_date or end > max_date:
            raise ValueError("Dates are out of range for available date.")

        df = df[(df.index >= start) & (df.index <= end)]
        x = df[" Close"].values
        dts = df.index

        super(SPDaily, self).__init__(x=x[None, None, :], dts=dts, **kwargs)
