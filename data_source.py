from typing import *
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool
import shutil
import numpy as np

from stochastic_classical_models import fbm, mrw, skewed_mrw, poisson_mu
from global_const import *


"""
TIME SERIES DATA classes
- stored as npz files (zipped archive)
- accessed and generated using Loader classes
- manipulated using TimeSeriesData classe

Notations
- R: number of realizations of a process
- S: number of synthesis of a model
- N: dimension of a process 
- T: number of times
"""


class TimeSeriesBase:
    """ A base class that stores generated or real world data. """
    def __init__(self, R, S, N, T, process_name, X):
        self.R = R
        self.S = S
        self.N = N
        self.T = T
        self.process_name = process_name

        self.X = X

    def describe(self):
        return self.process_name

    def __call__(self, r=None, s=None):
        if r is None and s is None:
            return self.X
        elif r is None:
            return self.X[:, s, :, :]
        elif s is None:
            return self.X[r, ...]
        return self.X[r, s, :, :]


class TimeSeriesNpzFile(TimeSeriesBase):
    def __init__(self, filepath=None, ld=None):
        self.filepath = filepath
        if ld is None:
            ld = np.load(filepath, allow_pickle=True)
        R, S, N, T = ld['R'], ld['S'], ld['N'], ld['T']
        process_name, X = ld['process_name'], ld['X']
        super(TimeSeriesNpzFile, self).__init__(R, S, N, T, process_name, X)

        # add other attributes
        for (key, value) in ld.items():
            if key not in ['R', 'S', 'N', 'T', 'process_name', 'x']:
                self.__dict__[key] = value


"""
LOADERS classes are here to access cached data 
"""


class ProcessDataLoader:
    def __init__(self, model_name: str, dirname: Optional[Union[str, Path]] = None):
        self.model_name = model_name
        self.dir_name = Path(dirname) or SYNTHETIC_DATA / self.model_name
        self.default_kwargs = None

        self.mkdir()

    def mkdir(self) -> None:
        self.dir_name.mkdir(parents=True, exist_ok=True)

    # kept for compatibility reason
    def dirpath_old(self, **kwargs) -> Path:
        def format_path(value):
            if isinstance(value, str):
                return f"_{value}"
            if isinstance(value, dict):
                return "".join([format_path(v) for (k, v) in value.items()])
            else:
                return f"_{value:.1e}"
        fname = self.model_name + format_path(kwargs)
        return self.dir_name / fname

    def dirpath(self, **kwargs) -> Path:
        def format_path(value):
            if isinstance(value, str):
                return f"_{value}"
            if isinstance(value, dict):
                return "".join([format_path(v) for (k, v) in value.items()])
            if isinstance(value, int):
                return f"_{value}"
            else:
                return f"_{value:.1e}"
        fname = self.model_name + format_path(kwargs)
        return self.dir_name / fname

    def generate_trajectory(self, **kwargs) -> np.ndarray:
        pass

    def worker(self, i: Any, **kwargs) -> None:
        np.random.seed(None)
        try:
            X = self.generate_trajectory(**kwargs)
            fname = f"{np.random.randint(1e7, 1e8)}.npy"
            np.save(str(kwargs['dirpath'] / fname), X[None, None, :, :])
        except ValueError as e:
            print(e)
            return

    def generate(self, dirpath, R_gen, **kwargs) -> dict:
        print(f"{self.model_name}: generating data.")
        kwargs_gen = {key: value for key, value in kwargs.items() if key != 'R'}

        # multiprocess generation
        pool = Pool(8)
        pool.map(partial(self.worker, **{**kwargs_gen, **{'dirpath': dirpath}}), [0] * R_gen)

        return kwargs

    def load(self, **kwargs) -> TimeSeriesNpzFile:
        full_kwargs = self.default_kwargs.copy()

        # add kwargs to default_args
        for (key, value) in kwargs.items():
            if key in self.default_kwargs:
                full_kwargs[key] = value
        dirpath = self.dirpath(**{key: value for (key, value) in full_kwargs.items() if key != 'R'})
        dirpath_old = self.dirpath_old(**{key: value for (key, value) in full_kwargs.items() if key != 'R'})
        if dirpath_old.is_dir():
            dirpath = dirpath_old
        if len(str(dirpath)) > 255:
            raise ValueError("Path is too long.")

        # generate if necessary
        dirpath.mkdir(exist_ok=True)
        R_available = len(list(dirpath.glob('*')))
        if R_available < full_kwargs['R']:
            self.generate(R_gen=full_kwargs['R'] - R_available, dirpath=dirpath, **full_kwargs)
        if len(list(dirpath.glob('*'))) < full_kwargs['R']:
            print(f"Incomplete generation {len(list(dirpath.glob('*')))}/{full_kwargs['R']}.")

        # return available realizations
        X = np.concatenate([np.load(str(fn)) for fn in dirpath.iterdir()])[:full_kwargs['R'], ...]
        full_kwargs['process_name'] = self.model_name
        full_kwargs['S'] = full_kwargs['N'] = 1
        full_kwargs['R'] = X.shape[0]
        full_kwargs['X'] = X
        return TimeSeriesNpzFile(ld=full_kwargs)

    def erase(self, **kwargs) -> None:
        full_kwargs = self.default_kwargs.copy()
        # add kwargs to default_args
        for (key, value) in kwargs.items():
            if key in self.default_kwargs:
                full_kwargs[key] = value
        shutil.rmtree(self.dirpath(**{key: value for (key, value) in full_kwargs.items() if key != 'R'}))


class PoissonLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(PoissonLoader, self).__init__('poisson', dirname)
        self.default_kwargs = OrderedDict({'R': 1, 'T': 2 ** 12, 'mu': 0.01, 'signed': False})

    def generate_trajectory(self, **kwargs):
        return poisson_mu(T=kwargs['T'], mu=kwargs['mu'], signed=kwargs['signed'])[None, :]


class FBmLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(FBmLoader, self).__init__('fBm', dirname)
        self.default_kwargs = OrderedDict({'R': 1, 'T': 2 ** 12, 'H': 0.5})

    def generate_trajectory(self, **kwargs):
        return fbm(R=1, T=kwargs['T'], H=kwargs['H'])


class MRWLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(MRWLoader, self).__init__('MRW', dirname)
        self.default_kwargs = OrderedDict({'R': 1, 'T': 2 ** 12, 'L': 2 ** 12, 'H': 0.5, 'lam': 0.1})

    def generate_trajectory(self, **kwargs):
        return mrw(R=1, T=kwargs['T'], L=kwargs['T'], H=kwargs['H'], lam=kwargs['lam'])


class SMRWLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(SMRWLoader, self).__init__('SMRW', dirname)
        self.default_kwargs = OrderedDict({'R': 1, 'T': 2 ** 12, 'L': 2 ** 12, 'dt': 1, 'H': 0.5, 'lam': 0.1,
                                           'K': 0.035, 'alpha': 0.23, 'beta': 0.5, 'gamma': 1 / (2**12) / 64})

    def generate_trajectory(self, **kwargs):
        return skewed_mrw(R=1, T=kwargs['T'], L=kwargs['L'], dt=kwargs['dt'], H=kwargs['H'], lam=kwargs['lam'],
                          K0=kwargs['K'], alpha=kwargs['alpha'], beta=kwargs['beta'], gamma=kwargs['gamma'])


class SynthesisLoader:
    """Loads the different trajectories from a synthesized directory."""
    def __init__(self, dir: str):
        self.dir = Path(dir)

    def load(self, key='x_synt', S=None, t1=None, t2=None):
        """Load all trajectories present in the directroy."""
        Xs = [np.load(str(fname))[key][:, t1:t2] if t1 is not None else np.load(str(fname))[key]
              for (s, fname) in enumerate(list(self.dir.iterdir())) if (s is None) or (s < S)]
        return np.stack(Xs)[None, :, :, :]  # R=1, S, N, T
