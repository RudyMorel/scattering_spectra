""" A class that implements the scale paths used in the scattering transform. """
from typing import List, Tuple, Dict
from itertools import product, chain
from collections import OrderedDict
import numpy as np
import torch
import pandas as pd

from scatspectra.utils import format_args_to_list


""" Notations
- sc_path or path: a scale path, tuple (j1, j2, ... jr)
- sc_idx or idx: paths are numbered by their scale path
- J: number of octaves
- Q1: number of wavelets per octave on first wavelet layer
- Q2: number of wavelets per octave on second wavelet layer
"""


class ScatteringShape:
    def __init__(self,
                 N: int,
                 n_scales: int,
                 A: int,
                 T: int) -> None:
        self.N = N
        self.n_scales = n_scales
        self.A = A
        self.T = T


class ScaleIndexer:
    """ Implements the scale paths used in the scattering transform. """

    def __init__(self,
                 r: int,
                 J: int | List[int],
                 Q: int | List[int],
                 strictly_increasing: bool = True) -> None:
        self.r = r
        self.J, self.Q = format_args_to_list(J, Q, n=r)
        self.strictly_increasing = strictly_increasing

        self.sc_paths = self._create_scale_paths()
        self.p_coding, self.p_decoding = self._construct_path_coding_dicts()
        self.sc_idces = self._create_scale_indices()

        self.low_pass_mask = self._compute_low_pass_mask()

    def JQ(self, r: int) -> int:
        """ Return the number of wavelet at layer r. """
        return self.J[r-1] * self.Q[r-1]

    def _admissible_scale_path(self, path: Tuple) -> bool:
        """ Tells if path j1, j2 ... j{r-1} jr is admissible. """

        def compare(i, j):
            return i < j if self.strictly_increasing else i <= j

        # only path shorter than r
        depth_ok = len(path) <= self.r

        # only paths with increasing scales
        scales_ok = all(
            compare(i//self.Q[o], j//self.Q[o+1])
            for o, (i, j) in enumerate(zip(path[:-1], path[1:]))
        )

        # if a low-pass filter is used, it must be on the last scale
        low_pass_ok = all(j < self.JQ(o+1) for o, j in enumerate(path[:-1]))

        return depth_ok and scales_ok and low_pass_ok

    def _create_scale_paths(self) -> List[np.ndarray]:
        """ The tables j1, j2 ... j{r-1} jr for every order r. """
        sc_paths = []
        for r in range(1, self.r + 1):
            sc_paths_r = np.array([
                p for p in product(*[range(self.JQ(o+1)+1) for o in range(r)])
                if self._admissible_scale_path(p)
            ])
            sc_paths.append(sc_paths_r)
        return sc_paths

    def _create_scale_indices(self) -> List[np.ndarray]:
        """ The scale indices numerating scale paths. """
        sc_idces = []
        for r in range(1, self.r + 1):
            sc_idces_r = np.array([
                self.path_to_idx(p) for p in self.sc_paths[r-1]
            ])
            sc_idces.append(sc_idces_r)
        return sc_idces

    def _construct_path_coding_dicts(self) -> Tuple[Dict[Tuple, int], Dict[int, Tuple]]:
        """ Construct the associations idx -> path and path -> idx. """

        coding = OrderedDict()
        coding[()] = 0
        for i, path in enumerate(list(chain.from_iterable(self.sc_paths))):
            coding[tuple(path)] = i

        decoding = OrderedDict({v: k for k, v in coding.items()})

        return coding, decoding

    def _compute_low_pass_mask(self) -> List[torch.Tensor]:
        """ Compute the low pass mask telling at each order which are the paths ending with a low pass filter. """
        return [
            torch.LongTensor(paths[:, -1]) == self.JQ(order+1)
            for order, paths in enumerate(self.sc_paths[:3])
        ]

    def get_all_paths(self) -> List[Tuple]:
        return list(self.p_coding.keys())

    def get_all_idces(self) -> List[int]:
        return list(self.p_decoding.keys())

    def path_to_idx(self, path: Tuple | List | np.ndarray) -> int:
        """ Return scale index i corresponding to path. """
        path = np.array(path)
        if len(path) > 0 and path[-1] == -1:
            i0 = np.argmax(path == -1)
            path = path[:i0]
        return self.p_coding[tuple(path)]

    def idx_to_path(self, idx: int, squeeze: bool = True) -> Tuple:
        """ Return scale path j1, ... j{r-1} jr corresponding to scale index i. """
        if idx == -1:
            return tuple()
        path = self.p_decoding[idx]
        if squeeze:
            return path
        return path + (pd.NA, ) * (self.r - len(path))

    def is_low_pass(self, path: Tuple | List | np.ndarray) -> bool:
        """ Determines if the path indexed by idx is ending with a low-pass. """
        if isinstance(path, (int, np.integer)):
            path = self.idx_to_path(int(path))
        return path[-1] >= self.JQ(self.order(path))

    def order(self, path: Tuple | List | np.ndarray) -> int:
        """ The scattering order of the path indexed by idx. """
        if isinstance(path, (int, np.integer)):
            path = self.idx_to_path(int(path))
        return len(path)
