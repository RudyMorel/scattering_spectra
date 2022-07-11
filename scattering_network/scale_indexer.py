from typing import *
from itertools import product, chain
from collections import OrderedDict
from functools import cmp_to_key
import numpy as np
import torch


class ScaleIndexer:
    # todo: check if we could get rid of this class, especially get rid of the scale indices
    def __init__(self, J: int, Q1: int, Q2: int, r_max: int):
        self.J, self.Q1, self.Q2, self.r_max = J, Q1, Q2, r_max

        self.p_idx = self.compute_p_idx()  # list[order] array
        self.p_coding, self.p_decoding = self.construct_path_coding_dicts()

        self.low_pass_mask = self.compute_low_pass_mask()  # list[order] array

        self.checks()

    def checks(self):
        # path and idx are in same order
        argsort = np.argsort(np.array(list(self.p_coding.values())))
        paths = list(self.p_coding.keys())
        def compare(t1, t2): return (len(t1) == len(t2) and t1 < t2) or (len(t1) < len(t2))  # order on tuples
        assert sorted(paths, key=cmp_to_key(compare)) == [paths[i] for i in argsort]

        # TODO: add r=3 checks
        for r in ([2] if self.r_max >= 2 else []):
            # when p_idx[r] is collapsed we obtain p_idx[r-1] without low_pass
            collapsed = np.unique(self.p_idx[r - 1][:, :-1], axis=0)
            previous_order = self.p_idx[r - 2][~self.low_pass_mask[r-2], :]
            assert np.all(collapsed == previous_order)

    def JQ(self, r=1) -> int:
        return self.J * {1: self.Q1, 2: self.Q2}[r]

    def condition(self, path) -> bool:
        """Tells if path j1, j2 ... j{r-1} jr is admissible."""
        return (len(path) <= self.r_max) and all(i < j for i, j in zip(path[:-1], path[1:]))

    def compute_p_idx(self) -> List[np.ndarray]:
        """The tables j1, j2 ... j{r-1} jr for every order r."""
        return [np.arange(self.JQ() + 1)[:, None]] + \
               [np.array([path for path in product(*(range(self.JQ() + 1),) * r) if self.condition(path)])
                for r in range(2, self.r_max + 1)]

    def construct_path_coding_dicts(self) -> Tuple[Dict[Tuple, int], Dict[int, Tuple]]:
        coding = OrderedDict({(): 0})
        coding = OrderedDict(coding, **{tuple(path): i for i, path in enumerate(list(chain.from_iterable(self.p_idx)))})
        decoding = OrderedDict({v: k for k, v in coding.items()})

        return coding, decoding

    def get_all_path(self) -> List[Tuple]:
        return list(self.p_coding.keys())

    def get_all_idx(self) -> List[int]:
        return list(self.p_decoding.keys())

    def path_to_idx(self, path: Union[Tuple, List, np.ndarray]) -> int:
        """Return global index i corresponding to path."""
        path = np.array(path)
        if len(path) > 0 and path[-1] == -1:
            i0 = np.argmax(path == -1)
            path = path[:i0]
        return self.p_coding[tuple(path)]

    def idx_to_path(self, idx: int, squeeze: Optional[bool] = True) -> Tuple[int]:
        """Return scale path j1, j2 ... j{r-1} jr corresponding to global index i."""
        if idx == -1:
            return tuple()
        if squeeze:
            return self.p_decoding[idx]
        return self.p_decoding[idx] + (-1, ) * (self.r_max - len(self.p_decoding[idx]))

    def is_low_pass(self, idx) -> bool:
        return self.idx_to_path(idx)[-1] >= self.JQ()

    def r(self, idx) -> int:
        return len(self.idx_to_path(idx))

    def is_max_path(self, idx) -> bool:
        return len(self.idx_to_path(idx)) == self.r_max

    def compute_low_pass_mask(self) -> List[torch.Tensor]:
        return [torch.LongTensor(paths[:, -1]) >= self.JQ() for paths in self.p_idx[:3]]
