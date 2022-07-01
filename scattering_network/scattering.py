from typing import *
from itertools import product, chain
from collections import OrderedDict, namedtuple
from functools import cmp_to_key
import numpy as np
import torch

import utils.complex_utils as cplx
from utils import multid_where_np
from scattering_network.module_chunk import SubModuleChunk
from scattering_network.model_output import IdxInfo, ModelOutput
from scattering_network.time_layers import Wavelet


class ScaleIndexer:
    # todo: check if we could get rid of this class, especially get rid of the scale indices
    def __init__(self, J: int, Q: int, r_max: int):
        self.J, self.Q, self.r_max = J, Q, r_max

        self.p_idx = self.compute_p_idx()  # list[order] array
        self.p_coding, self.p_decoding = self.construct_path_coding_dicts()

        self.conv_pair = self.compute_conv_pair()  # list[order] array
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
            # when p_idx[r] is collapsed we obtain p_idx[r-1]
            collapsed = np.unique(self.p_idx[r - 1][:, :-1], axis=0)
            previous_order = self.p_idx[r - 2][:-2, :]
            assert np.all(collapsed == previous_order)

        for r in [1, 2][:self.r_max]:
            # when conv_pairs (idx, jr) are deployed (j1,...,jr) we get p_idx
            deployed_left = np.array([np.append(self.idx_to_path(i), jr) for (i, jr) in self.conv_pair[r - 1]])
            if r == 1:
                deployed_left = deployed_left[:, 1:]
            assert np.all(deployed_left == self.p_idx[r - 1])

    def JQ(self) -> int:
        return self.J * self.Q

    def condition(self, path) -> bool:
        """Tells if path j1, j2 ... j{r-1} jr is admissible."""
        return (len(path) <= self.r_max) and all(i < j for i, j in zip(path[:-1], path[1:]))

    def compute_p_idx(self) -> List[np.ndarray]:
        """The tables j1, j2 ... j{r-1} jr for every order r."""
        return [np.arange(self.JQ() + 2)[:, None]] + \
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
        if isinstance(path, list) or (isinstance(path, np.ndarray) and path.ndim == 1):
            path = tuple(path)
        return self.p_coding[path]

    def idx_to_path(self, idx: int, squeeze: Optional[bool] = True) -> Tuple[int]:
        """Return scale path j1, j2 ... j{r-1} jr corresponding to global index i."""
        if squeeze:
            return self.p_decoding[idx]
        return self.p_decoding[idx] + (-1, ) * (self.r_max - len(self.p_decoding[idx]))

    def is_low_pass(self, idx) -> bool:
        return self.idx_to_path(idx)[-1] >= self.JQ()

    def r(self, idx) -> int:
        return len(self.idx_to_path(idx))

    def is_max_path(self, idx) -> bool:
        return len(self.idx_to_path(idx)) == self.r_max

    def compute_conv_pair(self) -> List[np.ndarray]:
        nbr = np.array([0, 0] + [p_idx.shape[0] for p_idx in self.p_idx[:-1]]).cumsum()
        return [np.array([(self.path_to_idx(path[:-1]) - nbr[r - 1], path[-1]) for path in self.p_idx[r - 1]])
                for r in range(1, self.r_max + 1)]

    def compute_low_pass_mask(self) -> List[torch.Tensor]:
        return [torch.LongTensor(paths[:, -1]) >= self.JQ() for paths in self.p_idx[:3]]


class Scattering(SubModuleChunk):
    def __init__(self, J, Q, r, T, A, N, wav_type, high_freq, rm_high, wav_norm, normalize):
        super(Scattering, self).__init__(require_normalization=False)
        self.J = J
        self.Q = Q
        self.r = r
        self.T = T
        self.A = A
        self.N = N

        self.W = Wavelet(T, J, Q, wav_type, high_freq, wav_norm)

        self.rm_high = rm_high
        self.W_no_HF = Wavelet(self.T, 1, 1, 'battle_lemarie', 0.5, 'l1')

        self.sc_idxer = ScaleIndexer(J, Q, r)

        self.normalize = normalize
        self.sigma = None

        # params
        self.channel_idx = []
        self.sc_pairing = []
        self.sc_out = []
        self.dx_param = []

    def resize(self, new_T) -> None:
        self.T = new_T
        self.W = Wavelet(new_T, self.J, self.Q, self.W.wav_type, self.W.high_freq, self.W.wav_norm, None)

    def external_surjection_aux(self, input_idx_info: Optional[NamedTuple] = None) -> List[NamedTuple]:
        """Return IdxInfo which depends on row to be computed."""
        idx_info_l = []
        out_columns = ['n', 'r', 'sc'] + [f'j{r}' for r in range(1, self.r + 1)] + ['a', 'lp']
        ns = range(self.N or 1) if input_idx_info is None else [input_idx_info[0]]
        for i, (n, sc, a) in enumerate(product(ns, self.sc_idxer.get_all_idx(), range(self.A or 1))):

            if sc == self.sc_idxer.JQ() + 1:
                continue

            row = [-1] * (5 + self.r)

            # n, r, sc
            row[:3] = n, self.sc_idxer.r(sc), sc

            # j1, j2, ..., jr
            row[3: 3 + self.sc_idxer.r(sc)] = self.sc_idxer.idx_to_path(sc)

            # a, lp
            row[-2:] = a, self.sc_idxer.is_low_pass(sc)

            idx_info_l.append(namedtuple('IdxInfo', out_columns)(*row))

        return idx_info_l

    def internal_surjection(self, row: NamedTuple) -> List[NamedTuple]:
        """From output_idx_info return the idxinfos that depend on output_idx_info to be computed."""
        n, r, sc, *js, a, lp = row
        out_columns = list(row._asdict().keys())

        # all scale paths that contain j1, j2, j3, ... , jr
        scs = [self.sc_idxer.path_to_idx(sc_path) for sc_path in self.sc_idxer.get_all_path()
               if len(sc_path) > r and sc_path[:r] == tuple(js[:r])]
        row_l = [(n, self.sc_idxer.r(sc), sc, *self.sc_idxer.idx_to_path(sc, squeeze=False),
                  a, self.sc_idxer.is_low_pass(sc)) for sc in scs]

        return [namedtuple('IdxInfo', out_columns)(*row) for row in row_l]

    def init_one_chunk(self, y: Optional[ModelOutput], output_idx_info: IdxInfo, i_chunk: int) -> None:
        channel_idx = []
        sc_pairing = []
        sc_out = []
        # dx_param = []

        # first order (r = 1)
        ns_j1s = np.unique(output_idx_info.to_array(['n', 'j1']), axis=0)
        channel_idx.append(ns_j1s[:, 0])
        dx_param = (ns_j1s[:, 1] == self.sc_idxer.JQ() + 1).astype(int)
        # sc_pairing.append(np.stack([ns_j1s[:, 1], np.clip(ns_j1s[:, 1], 0, self.sc_idxer.JQ())], axis=1))
        sc_pairing.append(np.stack([ns_j1s[:, 1], ns_j1s[:, 1]], axis=1))
        sc_out.append(multid_where_np(output_idx_info.reduce(r=1).to_array(['n', 'j1']), ns_j1s))

        # first order (r = 2)
        if output_idx_info.reduce(r=2).size() > 0:
            ns_j1s_j2s = np.unique(output_idx_info.to_array(['n', 'j1', 'j2']), axis=0)
            ns_j1s_j2s = ns_j1s_j2s[ns_j1s_j2s[:, -1] > -1]
            channel_idx.append(multid_where_np(ns_j1s_j2s[:, :2], ns_j1s))
            sc_pairing.append(np.stack([np.arange(ns_j1s_j2s.shape[0]), ns_j1s_j2s[:, 2]], axis=1))
            sc_out.append(multid_where_np(output_idx_info.reduce(r=2).to_array(['n', 'j1', 'j2']), ns_j1s_j2s))

        # second order (r = 3)
        if output_idx_info.reduce(r=3).size() > 0:
            ns_j1s_j2s_j3s = np.unique(output_idx_info.to_array(['n', 'j1', 'j2', 'j3']), axis=0)
            ns_j1s_j2s_j3s = ns_j1s_j2s_j3s[ns_j1s_j2s_j3s[:, -1] > -1]
            channel_idx.append(multid_where_np(ns_j1s_j2s_j3s[:, :3], ns_j1s_j2s))
            sc_pairing.append(np.stack([np.arange(ns_j1s_j2s_j3s.shape[0]), ns_j1s_j2s_j3s[:, 3]], axis=1))
            sc_out.append(multid_where_np(output_idx_info.reduce(r=3).to_array(['n', 'j1', 'j2', 'j3']), ns_j1s_j2s_j3s))

        self.channel_idx.append(channel_idx)
        self.sc_pairing.append(sc_pairing)
        self.sc_out.append(sc_out)
        self.dx_param.append(dx_param)

    def get_output_space_dim(self):
        return self.T * 2

    def clear_params(self) -> None:
        self.channel_idx = []
        self.sc_pairing = []
        self.sc_out = []

    def init_norm(self, sigma: torch.tensor) -> None:
        """Initialize the sigma(j) used to normalize first scale channel."""
        self.sigma = sigma

    def forward_chunk(self, x: torch.Tensor, i_chunk: int) -> ModelOutput:
        """
        Computes the representation B1x*psi_j, B2|B1x*psi_j1|*psi_j2.

        :param x: 1 x N x T x 2 tensor
        :param i_chunk:
        :return: Scattering output
        """
        idx_info = self.idx_info[i_chunk]

        y = x.unsqueeze(-3)
        sx = x.new_zeros(idx_info.size(), self.T, 2)

        if self.rm_high:
            y = self.W_no_HF(y.unsqueeze(-4), np.array([[0, 1]]))[..., 0, :, :, :]

        for r, c_idx, sc_pairing, sc_out in \
                zip(range(1, self.r + 1), self.channel_idx[i_chunk], self.sc_pairing[i_chunk], self.sc_out[i_chunk]):
            if r > 1:
                y = cplx.from_real(cplx.modulus(y))
                y = y[:, c_idx, ...]
            else:
                dy = torch.zeros_like(y)
                dy[..., 1:, :] = y[..., 1:, :] - y[..., :-1, :]
                y = torch.stack([y, dy], dim=1)[:, self.dx_param[i_chunk], c_idx, ...]

            # time layer
            y = self.W(y, sc_pairing)

            # normalize scale axis
            if r == 1 and self.normalize and self.sigma is None:
                band_pass = sc_pairing[:, -1] < self.sc_idxer.JQ()
                sigma = cplx.modulus(y[:, band_pass, ...]).pow(2.0).mean(-1).pow(0.5)[..., None, None]
                y[:, band_pass, ...] /= sigma
            if r == 1 and self.normalize and self.sigma is not None:
                y /= self.sigma[sc_pairing[:, 1]][None, :, None, None, None]

            sx[idx_info.where(r=r), :, :] = y[0, sc_out, 0, :, :]

        return ModelOutput(x=None, idx_info=idx_info, y=sx)
