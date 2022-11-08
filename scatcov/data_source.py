""" Data classes and functions. """
from typing import *
from pathlib import Path
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool
import shutil
import numpy as np

from scatcov.stochastic_classical_models import fbm, mrw, skewed_mrw, poisson_mu


""" TIME SERIES DATA classes
- stored as npz files (zipped archive)
- accessed and generated using Loader classes
- manipulated using TimeSeriesData classe

Notations
- S: number of syntheses
- B: number of batch (i.e. realizations of a process) used to average a representation
- N: number of data channels (N>1 : multivariate process)
- T: number of data samples (time samples)
"""


class TimeSeriesBase:
    """ A base class that stores generated or real world data. """
    def __init__(self, R, N, T, process_name, x):
        self.R = R
        self.N = N
        self.T = T
        self.process_name = process_name

        self.x = x

    def describe(self):
        return self.process_name

    def __call__(self, r=None):
        if r is None:
            return self.x
        return self.x[r, :, :]


class TimeSeriesNpzFile(TimeSeriesBase):
    """ A time-series class obtained from loading a .npz file. """
    def __init__(self, filepath=None):
        self.filepath = filepath
        ld = np.load(filepath, allow_pickle=True)
        R, N, T = ld['R'], ld['N'], ld['T']
        process_name, x = ld['process_name'], ld['x']
        super(TimeSeriesNpzFile, self).__init__(R, N, T, process_name, x)

        # add other attributes
        for (key, value) in ld.items():
            if key not in ['R', 'N', 'T', 'process_name', 'x']:
                self.__dict__[key] = value


class TimeSeriesDir(TimeSeriesBase):
    """ A time-series class obtained from a directory of trajectories. """
    def __init__(self, dirpath: Path, R: Optional[int] = None):
        x = np.concatenate([np.load(str(fn)) for fn in dirpath.iterdir()])
        if R is not None:
            x = x[:R, :, :]
        R, N, T = x.shape
        super(TimeSeriesDir, self).__init__(R, N, T, dirpath.name[:5], x)


"""
LOADER classes create and access cached data 
"""


class ProcessDataLoader:
    """ Base process data loader class. """
    def __init__(self, model_name: str, dirname: Optional[Union[str, Path]] = None, num_workers: Optional[int] = 1):
        self.model_name = model_name
        self.dir_name = Path(__file__).parents[0] / '_cached_dir' if dirname is None else Path(dirname)
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
            x = self.generate_trajectory(**kwargs)
            fname = f"{np.random.randint(1e7, 1e8)}.npy"
            np.save(str(kwargs['dirpath'] / fname), x)
            print(f"Saved: {kwargs['dirpath'].name}/{fname}")
        except ValueError as e:
            print(e)
            return

    def generate(self, dirpath: Path, n_jobs: int, **kwargs) -> dict:
        """ Performs a cached generation saving into dirpath. """
        print(f"{self.model_name}: generating data.")
        kwargs_gen = {key: value for key, value in kwargs.items() if key != 'n_files'}

        # multiprocess generation
        pool = Pool(self.num_workers)
        pool.map(partial(self.worker, **{**kwargs_gen, **{'dirpath': dirpath}}), np.arange(n_jobs))

        return kwargs

    def load(self, R=1, **kwargs) -> TimeSeriesDir:
        """ Loads the data required, generating it if not present in the cache. """
        full_kwargs = self.default_kwargs.copy()

        # add kwargs to default_args
        for (key, value) in kwargs.items():
            if key in self.default_kwargs or self.default_kwargs == {}:
                full_kwargs[key] = value
        dirpath = self.dirpath(**full_kwargs)
        if len(str(dirpath)) > 255:
            raise ValueError(f"Path is too long ({len(str(dirpath))} > 250).")

        # generate if necessary
        dirpath.mkdir(exist_ok=True)
        nfiles_available = len(list(dirpath.glob('*')))
        if nfiles_available < kwargs['n_files']:
            print(f"Data saving dir: {dirpath.name}")
            self.generate(n_jobs=kwargs['n_files']-nfiles_available, dirpath=dirpath, **full_kwargs)
        if len(list(dirpath.glob('*'))) < kwargs['n_files']:
            print(f"Incomplete generation {len(list(dirpath.glob('*')))} files/{kwargs['n_files']}.")

        # return available realizations
        print(f"Saved: {dirpath.name}")
        return TimeSeriesDir(dirpath=dirpath, R=R)

    def erase(self, **kwargs) -> None:
        """ Erase specified data if present in the cache. """
        full_kwargs = self.default_kwargs.copy()
        # add kwargs to default_args
        for (key, value) in kwargs.items():
            if key in self.default_kwargs:
                full_kwargs[key] = value
        shutil.rmtree(self.dirpath(**{key: value for (key, value) in full_kwargs.items() if key != 'n_files'}))


class PoissonLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(PoissonLoader, self).__init__('poisson', dirname)
        self.default_kwargs = OrderedDict({'T': 2 ** 12, 'mu': 0.01, 'signed': False})

    def generate_trajectory(self, **kwargs):
        return poisson_mu(T=kwargs['T'], mu=kwargs['mu'], signed=kwargs['signed'])[None, None, :]


class FBmLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(FBmLoader, self).__init__('fBm', dirname)
        self.default_kwargs = OrderedDict({'T': 2 ** 12, 'H': 0.5})

    def generate_trajectory(self, **kwargs):
        return fbm(R=1, T=kwargs['T'], H=kwargs['H'])[None, :]


class MRWLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(MRWLoader, self).__init__('MRW', dirname)
        self.default_kwargs = OrderedDict({'T': 2 ** 12, 'L': None, 'H': 0.5, 'lam': 0.1})

    def generate_trajectory(self, **kwargs):
        L = kwargs['L'] or kwargs['T']
        return mrw(R=1, T=kwargs['T'], L=L, H=kwargs['H'], lam=kwargs['lam'])[None, :]


class SMRWLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(SMRWLoader, self).__init__('SMRW', dirname)
        self.default_kwargs = OrderedDict({'T': 2 ** 12, 'L': None, 'dt': 1, 'H': 0.5, 'lam': 0.1,
                                           'K': 0.035, 'alpha': 0.23, 'beta': 0.5, 'gamma': 1 / (2**12) / 64})

    def generate_trajectory(self, **kwargs):
        L = kwargs['L'] or kwargs['T']
        return skewed_mrw(R=1, T=kwargs['T'], L=L, dt=kwargs['dt'], H=kwargs['H'], lam=kwargs['lam'],
                          K0=kwargs['K'], alpha=kwargs['alpha'], beta=kwargs['beta'], gamma=kwargs['gamma'])[None, :]
