""" Data classes and functions. """
from typing import *
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool
import shutil
import numpy as np

from stochastic_classical_models import fbm, mrw, skewed_mrw, poisson_mu
from global_const import *


""" TIME SERIES DATA classes
- stored as npz files (zipped archive)
- accessed and generated using Loader classes
- manipulated using TimeSeriesData classe

Notations
- B: number of batch (i.e. realizations of a process)
- N: number of data channels (N>1 : multivariate process)
- T: number of data samples (time samples)
"""


class TimeSeriesBase:
    """ A base class that stores generated or real world data. """
    def __init__(self, B, N, T, process_name, X):
        self.B = B
        self.N = N
        self.T = T
        self.process_name = process_name

        self.X = X

    def describe(self):
        return self.process_name

    def __call__(self, r=None):
        if r is None:
            return self.X
        return self.X[r, :, :]


class TimeSeriesNpzFile(TimeSeriesBase):
    """ A time-series class obtained from loading a .npz file. """
    def __init__(self, filepath=None):
        self.filepath = filepath
        ld = np.load(filepath, allow_pickle=True)
        B, N, T = ld['B'], ld['N'], ld['T']
        process_name, X = ld['process_name'], ld['X']
        super(TimeSeriesNpzFile, self).__init__(B, N, T, process_name, X)

        # add other attributes
        for (key, value) in ld.items():
            if key not in ['B', 'S', 'N', 'T', 'process_name', 'x']:
                self.__dict__[key] = value


class TimeSeriesDir(TimeSeriesBase):
    """ A time-series class obtained from a directory of trajectories. """
    def __init__(self, dirpath: Path, B: Optional[int] = None):
        X = np.concatenate([np.load(str(fn)) for fn in dirpath.iterdir()])
        if B is not None:
            X = X[:B, :, :]
        B, N, T = X.shape
        super(TimeSeriesDir, self).__init__(B, N, T, dirpath.name[:5], X)


"""
LOADER classes create and access cached data 
"""


class ProcessDataLoader:
    """ Base process data loader class. """
    def __init__(self, model_name: str, dirname: Optional[Union[str, Path]] = None, num_workers: Optional[int] = 1):
        self.model_name = model_name
        self.dir_name = Path(__file__).parents[0] / 'cached_dir' if dirname is None else Path(dirname)
        self.num_workers = num_workers
        self.default_kwargs = None

        self.mkdir()

    def mkdir(self) -> None:
        self.dir_name.mkdir(parents=True, exist_ok=True)

    def dirpath(self, **kwargs) -> Path:
        def format_path(key, value):
            if isinstance(value, dict):
                return "".join([format_path(k, v) for (k, v) in value.items()])
            elif value is None:
                return f"_none"
            elif isinstance(value, str):
                return f"_{key[:2]}_{value}"
            elif isinstance(value, int):
                return f"_{key[:2]}_{value}"
            elif isinstance(value, float):
                return f"_{key[:2]}_{value:.1e}"
            elif isinstance(value, bool):
                return f"_{key[:2]}_{int(value)}"
            else:
                return ''
        fname = (self.model_name + format_path(None, kwargs)).replace('.', '_').replace('-', '_')
        return self.dir_name / fname

    def generate_trajectory(self, **kwargs) -> np.ndarray:
        pass

    def worker(self, i: Any, **kwargs) -> None:
        np.random.seed(None)
        try:
            X = self.generate_trajectory(**kwargs)
            fname = f"{np.random.randint(1e7, 1e8)}.npy"
            np.save(str(kwargs['dirpath'] / fname), X[None, :, :])
        except ValueError as e:
            print(e)
            return

    def generate(self, dirpath: Path, S_gen: int, **kwargs) -> dict:
        """ Performs a cached generation saving into dirpath. """
        print(f"{self.model_name}: generating data.")
        kwargs_gen = {key: value for key, value in kwargs.items() if key != 'B'}

        # multiprocess generation
        pool = Pool(self.num_workers)
        pool.map(partial(self.worker, **{**kwargs_gen, **{'dirpath': dirpath}}), np.arange(S_gen))

        return kwargs

    def load(self, **kwargs) -> TimeSeriesDir:
        """ Loads the data required, generating it if not present in the cache. """
        full_kwargs = self.default_kwargs.copy()

        # add kwargs to default_args
        for (key, value) in kwargs.items():
            if key in self.default_kwargs or self.default_kwargs == {}:
                full_kwargs[key] = value
        dirpath = self.dirpath(**{key: value for (key, value) in full_kwargs.items() if key != 'B'})
        if len(str(dirpath)) > 255:
            raise ValueError(f"Path is too long ({len(str(dirpath))} > 250).")

        # generate if necessary
        dirpath.mkdir(exist_ok=True)
        R_available = len(list(dirpath.glob('*')))
        if R_available < full_kwargs['B']:
            print(f"Data saving dir: {dirpath.name}")
            self.generate(S_gen=full_kwargs['B'] - R_available, dirpath=dirpath, **full_kwargs)
        if len(list(dirpath.glob('*'))) < full_kwargs['B']:
            print(f"Incomplete generation {len(list(dirpath.glob('*')))}/{full_kwargs['B']}.")

        # return available realizations
        print(f"Saved: {dirpath.name}")
        return TimeSeriesDir(dirpath=dirpath, B=full_kwargs['B'])

    def erase(self, **kwargs) -> None:
        """ Erase specified data if present in the cache. """
        full_kwargs = self.default_kwargs.copy()
        # add kwargs to default_args
        for (key, value) in kwargs.items():
            if key in self.default_kwargs:
                full_kwargs[key] = value
        shutil.rmtree(self.dirpath(**{key: value for (key, value) in full_kwargs.items() if key != 'B'}))


class PoissonLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(PoissonLoader, self).__init__('poisson', dirname)
        self.default_kwargs = OrderedDict({'B': 1, 'T': 2 ** 12, 'mu': 0.01, 'signed': False})

    def generate_trajectory(self, **kwargs):
        return poisson_mu(T=kwargs['T'], mu=kwargs['mu'], signed=kwargs['signed'])[None, :]


class FBmLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(FBmLoader, self).__init__('fBm', dirname)
        self.default_kwargs = OrderedDict({'B': 1, 'T': 2 ** 12, 'H': 0.5})

    def generate_trajectory(self, **kwargs):
        return fbm(R=1, T=kwargs['T'], H=kwargs['H'])


class MRWLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(MRWLoader, self).__init__('MRW', dirname)
        self.default_kwargs = OrderedDict({'B': 1, 'T': 2 ** 12, 'L': None, 'H': 0.5, 'lam': 0.1})

    def generate_trajectory(self, **kwargs):
        L = kwargs['L'] or kwargs['T']
        return mrw(R=1, T=kwargs['T'], L=L, H=kwargs['H'], lam=kwargs['lam'])


class SMRWLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(SMRWLoader, self).__init__('SMRW', dirname)
        self.default_kwargs = OrderedDict({'B': 1, 'T': 2 ** 12, 'L': None, 'dt': 1, 'H': 0.5, 'lam': 0.1,
                                           'K': 0.035, 'alpha': 0.23, 'beta': 0.5, 'gamma': 1 / (2**12) / 64})

    def generate_trajectory(self, **kwargs):
        L = kwargs['L'] or kwargs['T']
        return skewed_mrw(R=1, T=kwargs['T'], L=L, dt=kwargs['dt'], H=kwargs['H'], lam=kwargs['lam'],
                          K0=kwargs['K'], alpha=kwargs['alpha'], beta=kwargs['beta'], gamma=kwargs['gamma'])
