from typing import *
from itertools import product
from collections import namedtuple
import torch

from scattering_network.module_chunk import SubModuleChunk
from scattering_network.scale_indexer import ScaleIndexer
from scattering_network.described_tensor import Description, DescribedTensor
from utils import multid_where_np, multid_where
import utils.complex_utils as cplx


class Marginal(SubModuleChunk):
    def __init__(self, qs: List[float]):
        super(Marginal, self).__init__(require_normalization=False)
        self.m_types = ['m00']

        self.qs = qs

        # params
        self.masks = []

    def external_surjection_aux(self, row: NamedTuple) -> List[NamedTuple]:
        """Return IdxInfo which depends on row to be computed."""
        out_columns = row._fields + ('q', 'm_type')

        def extend(row, q): return row + (q, 'm00')

        output_idxinfo_l = [namedtuple('IdxInfo', out_columns)(*extend(row, q)) for q in self.qs]

        return output_idxinfo_l

    def internal_surjection(self, row: NamedTuple) -> List[NamedTuple]:
        """From output_idx_info return the idxinfos that depend on output_idx_info to be computed."""
        return []

    def init_one_chunk(self, input: DescribedTensor, output_idx_info: Description, i_chunk: int) -> None:
        masks = []

        for q in self.qs:
            mask_q = multid_where_np(output_idx_info.reduce(q=q).drop(columns=['q', 'm_type']).values,
                                     input.idx_info.values)
            masks.append(mask_q)

        self.masks.append(masks)

    def clear_params(self) -> None:
        self.masks = []

    def get_output_space_dim(self):
        return 2

    def forward_chunk(self, x: torch.Tensor, i_chunk: int) -> DescribedTensor:
        """Computes E[SX] and E[SX SX^*]."""
        idx_info = self.idx_info[i_chunk]
        mask_qs = self.masks[i_chunk]
        y = x.new_zeros(idx_info.size(), 2)

        for q, mask_q in zip(self.qs, mask_qs):
            y[idx_info.where(q=q), 0] = cplx.modulus(x[mask_q, ...]).pow(q).mean(-1)

        return DescribedTensor(x=None, y=y, idx_info=idx_info)


class Cov(SubModuleChunk):
    """Compute order 1 and order 2 moments on input Sx: E[SX], E[SX SX^*]."""
    def __init__(self, N: int, sc_idxer: ScaleIndexer, m_types: Optional[List[str]] = None):
        super(Cov, self).__init__(require_normalization=False)
        self.N = N
        self.sc_idxer = sc_idxer

        possible_m_types = [f'm{r1}{r2}' for (r1, r2) in
                            product(range(self.sc_idxer.r_max), range(self.sc_idxer.r_max)) if r1 >= r2]
        desired_m_types = m_types or possible_m_types
        self.m_types = [m for m in desired_m_types if m in possible_m_types]

        # params
        self.masks = []

    def external_surjection_aux(self, row: NamedTuple) -> List[NamedTuple]:
        n1, r, sc, *js, a, low = row
        out_columns = ['n1', 'n1p', 'q', 'r', 'rp', 'sc', 'scp'] + \
                      [f'j{r}' for r in range(1, self.sc_idxer.r_max + 1)] + \
                      [f'jp{r}' for r in range(1, self.sc_idxer.r_max + 1)] + \
                      ['a', 'ap', 're', 'low', 'm_type']

        output_idxinfo_l = []

        # E[SX]
        if 'm00' in self.m_types and low:
            output_idxinfo_l.append((n1, -1, 1, max(1, r-1), -1, sc, -1, *js, *(-1,)*len(js), a, -1, low, r==1, 'm00'))

        # E[SX SX^*]
        for scp in self.sc_idxer.get_all_idx():
            path = self.sc_idxer.idx_to_path(sc)
            path_p = self.sc_idxer.idx_to_path(scp)
            rp = len(path_p)

            (scl, scr) = (sc, scp) if r > rp or (r == rp and path >= path_p) else (scp, sc)
            jl, jr = self.sc_idxer.idx_to_path(scl, squeeze=False), self.sc_idxer.idx_to_path(scr, squeeze=False)
            rl, rr = self.sc_idxer.r(scl), self.sc_idxer.r(scr)

            if path[-1] != path_p[-1] or f'm{rl-1}{rr-1}' not in self.m_types:
                continue

            output_idxinfo_l.append((n1, n1, 2, rl, rr, scl, scr, *jl, *jr, a, a, low or scl == scr, low, f'm{rl-1}{rr-1}'))

        return [namedtuple('IdxInfo', out_columns)(*row) for row in output_idxinfo_l]

    def internal_surjection(self, row: NamedTuple) -> List[NamedTuple]:
        """From output_idx_info return the idxinfos that depend on output_idx_info to be computed."""
        return []

    def init_one_chunk(self, input: DescribedTensor, output_idx_info: Description, i_chunk: int) -> None:
        """Init the parameters of the model required to compute output_idx_info from input_idx_info."""
        channel_scale = input.idx_info.to_array(['n1', 'sc'])

        # mask q = 1: E[SX]
        mask_q1 = multid_where_np(output_idx_info.reduce(q=1).to_array(['n1', 'sc']), channel_scale)

        # mask q = 2: E[SX SX^*]
        mask_q2_l = multid_where_np(output_idx_info.reduce(q=2).to_array(['n1', 'sc']), channel_scale)
        mask_q2_r = multid_where_np(output_idx_info.reduce(q=2).to_array(['n1p', 'scp']), channel_scale)

        self.masks.append((mask_q1, (mask_q2_l, mask_q2_r)))

    def get_output_space_dim(self):
        return 2

    def clear_params(self) -> None:
        self.masks = []

    def forward_chunk(self, x: torch.Tensor, i_chunk: int) -> DescribedTensor:
        """Computes E[SX] and E[SX SX^*]."""
        idx_info = self.idx_info[i_chunk]
        mask_q1, masks_q2 = self.masks[i_chunk]
        y = x.new_zeros((idx_info.size(), 2))

        # q = 1: E[SX]
        y[idx_info.where(q=1), ...] = x[mask_q1, :, :].mean(-2)

        # q = 2: E[SX SX^*]
        zl = x[masks_q2[0], :, :]
        zr = cplx.conjugate(x[masks_q2[1], :, :])
        y[idx_info.where(q=2), ...] = cplx.mul(zl, zr).mean(-2)

        return DescribedTensor(x=None, y=y, idx_info=idx_info)


class CovStat(SubModuleChunk):
    """
    Representation for wide-sense self-similar processes.
    Moments:
        - m00  :   E|X*psi_j|, E|X*psi_j|^2
        - m10  :   E^-1 E[|X*psi_{j-alpha}|*psi_j X*psi_j]
        - m11  :   E^-1 E[|x*psi_j|*psi_{j-beta} |x*psi{j-alpha}|*psi_{j-beta}^*]
    """

    def __init__(self, JQ: int, m_types: Optional[List[str]] = None):
        super(CovStat, self).__init__(require_normalization=False)
        self.JQ = JQ

        possible_m_types = ['m00', 'm10', 'm11']
        desired_m_types = m_types or possible_m_types
        self.m_types = [m for m in desired_m_types if m in possible_m_types]

    def external_surjection_aux(self, row: NamedTuple) -> List[NamedTuple]:
        """Return IdxInfo which depends on row to be computed."""
        n1, n2, q, r1, r2, sc1, sc2, j1, j2, jp1, jp2, a1, a2, re, low, m_type = row
        out_columns = ['n1', 'n2', 'q', 'r1', 'r2', 'j', 'alpha', 'beta', 'a1', 'a2', 're', 'low', 'm_type']
        row_l = []
        # todo: replace beta trivial from 0 to a nan value for example, in scat too
        if m_type == 'm00' and 'm00' in self.m_types:
            row_l.append((n1, n2, q, r1, r2, j1, 0, 0, a1, a2, re, low, m_type))
        if m_type == 'm10' and 'm10' in self.m_types:
            j, alpha = (j1, 0) if low else (-1, jp1 - j1)
            row_l.append((n1, n2, q, r1, r2, j, alpha, 0, a1, a2, low, low, 'm10'))
        if m_type == 'm11' and 'm11' in self.m_types:
            alpha, beta = j1 - jp1, (0 if low else j1 - j2)
            row_l.append((n1, n2, q, r1, r2, j1 if low else -1, alpha, beta, a1, a2, alpha == 0 or low, low, 'm11'))

        return [namedtuple('IdxInfo', out_columns)(*row) for row in row_l]

    def internal_surjection(self, row: NamedTuple) -> List[NamedTuple]:
        return []

    def construct_proj(self, input: DescribedTensor, output_idx_info: Description) -> torch.tensor:
        input_idx_info = input.idx_info
        output_idx_info_iter = list(output_idx_info.iter_tuple())

        # init projector
        proj = torch.zeros(output_idx_info.size(), input_idx_info.size(), dtype=torch.float64)
        for i, in_row in enumerate(input_idx_info.iter_tuple()):
            out_rows = self.external_surjection_aux(in_row)
            mask = multid_where(out_rows, output_idx_info_iter)
            proj[mask, i] = 1.0

        # to get average along j and not sum
        proj /= proj.sum(-1, keepdim=True)

        assert (proj.sum(-1) == 0).sum() == 0

        return proj

    def init_one_chunk(self, input: DescribedTensor, output_idx_info: Description, i_chunk: int) -> None:
        """Init the parameters of the model required to compute output_idx_info from input_idx_info."""
        self.register_buffer(f'proj_{i_chunk}', self.construct_proj(input, output_idx_info))

    def get_output_space_dim(self):
        return 2

    def forward_chunk(self, x: torch.tensor, i_chunk: int) -> DescribedTensor:
        """Computes moments."""
        idx_info = self.idx_info[i_chunk]

        proj = self.state_dict()[f'proj_{i_chunk}']

        moments = proj @ x

        return DescribedTensor(x=None, idx_info=idx_info, y=moments)
