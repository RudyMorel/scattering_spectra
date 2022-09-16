""" Moments to be used on top of a scattering transform. """
from typing import *
from itertools import product
from collections import namedtuple
import torch

from scatcov.scattering_network.module_chunk import SubModuleChunk
from scatcov.scattering_network.scale_indexer import ScaleIndexer
from scatcov.scattering_network.described_tensor import Description, DescribedTensor
from scatcov.utils import multid_where_np, multid_where
import scatcov.utils.complex_utils as cplx


class Marginal(SubModuleChunk):
    """ Compute per channel order q moments. """
    def __init__(self, qs: List[float]):
        super(Marginal, self).__init__(init_with_input=False)
        self.m_types = ['m00']

        self.qs = qs

        # params
        self.masks = []

    def external_surjection_aux(self, input_descri: NamedTuple) -> List[NamedTuple]:
        """ Return description that can be computed on input_descri. """
        out_columns = input_descri._fields + ('q', 'm_type')

        def extend(row, q): return row + (q, 'm00')

        output_descri_l = [namedtuple('Descri', out_columns)(*extend(input_descri, q)) for q in self.qs]

        return output_descri_l

    def internal_surjection(self, output_descri_row: NamedTuple) -> List[NamedTuple]:
        """ Return rows that can be computed on output_descri_row. """
        return []

    def init_one_chunk(self, input: DescribedTensor, output_descri: Description, i_chunk: int) -> None:
        """ Init the parameters of the model required to compute output_descri from input. """
        masks = []

        for q in self.qs:
            mask_q = multid_where_np(output_descri.reduce(q=q).drop(columns=['q', 'm_type']).values,
                                     input.descri.values)
            masks.append(mask_q)

        self.masks.append(masks)

    def clear_params(self) -> None:
        self.masks = []

    def get_output_space_dim(self):
        return 2

    def forward_chunk(self, x: torch.Tensor, i_chunk: int) -> DescribedTensor:
        """ Computes E[|SX|^q]. """
        descri = self.descri[i_chunk]
        mask_qs = self.masks[i_chunk]
        y = x.new_zeros(x.shape[0], descri.size(), 1, 2)

        for q, mask_q in zip(self.qs, mask_qs):
            y[:, descri.where(q=q), 0, 0] = cplx.modulus(x[:, mask_q, ...]).pow(q).mean(-1)

        return DescribedTensor(x=None, y=y, descri=descri)


class Cov(SubModuleChunk):
    """ Compute order 1 (per channel) and order 2 (per and cross channel) moments on SX: E[SX], E[SX SX^*]. """
    def __init__(self, N: int, sc_idxer: ScaleIndexer, m_types: Optional[List[str]] = None):
        super(Cov, self).__init__(init_with_input=False)
        self.N = N
        self.sc_idxer = sc_idxer

        possible_m_types = [f'm{r1}{r2}' for (r1, r2) in
                            product(range(self.sc_idxer.r_max), range(self.sc_idxer.r_max)) if r1 >= r2]
        desired_m_types = m_types or possible_m_types
        self.m_types = [m for m in desired_m_types if m in possible_m_types]

        # params
        self.masks = []

    def external_surjection_aux(self, row: NamedTuple) -> List[NamedTuple]:
        """ Return description that can be computed on input_descri. """
        n1, r, sc, *js, a, low = row
        out_columns = ['n1', 'n1p', 'q', 'r', 'rp', 'sc', 'scp'] + \
                      [f'j{r}' for r in range(1, self.sc_idxer.r_max + 1)] + \
                      [f'jp{r}' for r in range(1, self.sc_idxer.r_max + 1)] + \
                      ['a', 'ap', 're', 'low', 'm_type']

        output_descri_l = []

        # E[SX]
        if 'm00' in self.m_types and low:
            output_descri_l.append((n1, n1, 1, max(1, r-1), -1, sc, -1, *js, *(-1,)*len(js), a, -1, low, r==1, 'm00'))

        # E[SX SX^*]
        for scp in self.sc_idxer.get_all_idx():
            # scale interactions
            path = self.sc_idxer.idx_to_path(sc)
            path_p = self.sc_idxer.idx_to_path(scp)
            rp = len(path_p)

            (scl, scr) = (sc, scp) if r > rp or (r == rp and path >= path_p) else (scp, sc)
            jl, jr = self.sc_idxer.idx_to_path(scl, squeeze=False), self.sc_idxer.idx_to_path(scr, squeeze=False)
            rl, rr = self.sc_idxer.r(scl), self.sc_idxer.r(scr)
            ql, qr = self.sc_idxer.Qs[rl-1], self.sc_idxer.Qs[rr-1]

            if f'm{rl-1}{rr-1}' not in self.m_types:
                continue

            # correlate low pass with low pass or band pass with band pass and nothing else
            if (self.sc_idxer.is_low_pass(sc) and not self.sc_idxer.is_low_pass(scp)) or \
                    (not self.sc_idxer.is_low_pass(sc) and self.sc_idxer.is_low_pass(scp)):
                continue

            # only consider wavelets with non-negligibale overlapping support in Fourier
            if abs(path[-1] / ql - path_p[-1] / qr) >= 1:
                continue

            # channel interactions
            for n1p in range(n1, self.N):
                if (rr == rl == 1) and n1p < n1:
                    continue
                out_descri = (n1, n1p, 2, rl, rr, scl, scr, *jl, *jr, a, a, low or scl == scr, low, f'm{rl-1}{rr-1}')
                output_descri_l.append(out_descri)

        return [namedtuple('Description', out_columns)(*row) for row in output_descri_l]

    def internal_surjection(self, output_descri_row: NamedTuple) -> List[NamedTuple]:
        """ Return rows that can be computed on output_descri_row. """
        return []

    def init_one_chunk(self, input: DescribedTensor, output_descri: Description, i_chunk: int) -> None:
        """ Init the parameters of the model required to compute output_descri from input. """
        channel_scale = input.descri.to_array(['n1', 'sc'])

        # mask q = 1: E[SX]
        mask_q1 = multid_where_np(output_descri.reduce(q=1).to_array(['n1', 'sc']), channel_scale)

        # mask q = 2: E[SX SX^*]
        mask_q2_l = multid_where_np(output_descri.reduce(q=2).to_array(['n1', 'sc']), channel_scale)
        mask_q2_r = multid_where_np(output_descri.reduce(q=2).to_array(['n1p', 'scp']), channel_scale)

        self.masks.append((mask_q1, (mask_q2_l, mask_q2_r)))

    def get_output_space_dim(self):
        return 2

    def clear_params(self) -> None:
        self.masks = []

    def forward_chunk(self, x: torch.Tensor, i_chunk: int) -> DescribedTensor:
        """ Computes E[SX] and E[SX SX^*]. """
        descri = self.descri[i_chunk]
        mask_q1, masks_q2 = self.masks[i_chunk]
        y = x.new_zeros((x.shape[0], descri.size(), 1, 2))

        # q = 1: E[SX]
        y[:, descri.where(q=1), 0, :] = x[:, mask_q1, :, :].mean(-2)

        # q = 2: E[SX SX^*]
        zl = x[:, masks_q2[0], :, :]
        zr = cplx.conjugate(x[:, masks_q2[1], :, :])
        y[:, descri.where(q=2), 0, :] = cplx.mul(zl, zr).mean(-2)

        return DescribedTensor(x=None, y=y, descri=descri)


class CovStat(SubModuleChunk):
    """ Reduced representation by making covariances invariant to scaling.
    Moments:
        - m00  :   E|X*psi_j|, E|X*psi_j|^2
        - m10  :   E^-1 E[|X*psi_{j-a}|*psi_j X*psi_j]
        - m11  :   E^-1 E[|x*psi_j|*psi_{j-b} |x*psi{j-a}|*psi_{j-b}^*]
    """
    def __init__(self, JQ: int, m_types: Optional[List[str]] = None):
        super(CovStat, self).__init__(init_with_input=False)
        self.JQ = JQ

        possible_m_types = ['m00', 'm10', 'm11']
        desired_m_types = m_types or possible_m_types
        self.m_types = [m for m in desired_m_types if m in possible_m_types]

    def external_surjection_aux(self, row: NamedTuple) -> List[NamedTuple]:
        """ Return description that can be computed on input_descri. """
        n1, n1p, q, r1, rp1, sc, scp, j1, j2, jp1, jp2, a1, ap1, re, low, m_type = row
        out_columns = ['n1', 'n1p', 'q', 'r', 'rp', 'j', 'a', 'b', 're', 'low', 'm_type']
        row_l = []
        if m_type == 'm00' and 'm00' in self.m_types:
            row_l.append((n1, n1p, q, r1, rp1, j1, 0, 0, re, low, m_type))
        if m_type == 'm10' and 'm10' in self.m_types:
            j, a = (j1, 0) if low else (-1, jp1 - j1)
            row_l.append((n1, n1p, q, r1, rp1, j, a, 0, low, low, 'm10'))
        if m_type == 'm11' and 'm11' in self.m_types:
            a, b = j1 - jp1, (0 if low else j1 - j2)
            row_l.append((n1, n1p, q, r1, rp1, j1 if low else -1, a, b, a == 0 or low, low, 'm11'))

        return [namedtuple('Description', out_columns)(*row) for row in row_l]

    def internal_surjection(self, output_descri_row: NamedTuple) -> List[NamedTuple]:
        """ Return rows that can be computed on output_descri_row. """
        return []

    def construct_proj(self, input: DescribedTensor, output_descri: Description) -> torch.tensor:
        """ Construct the projector A that performs the average on scale. """
        input_descri = input.descri
        output_descri_iter = list(output_descri.iter_tuple())

        # init projector
        proj = torch.zeros(output_descri.size(), input_descri.size(), dtype=torch.float64)
        for i, in_row in enumerate(input_descri.iter_tuple()):
            out_rows = self.external_surjection_aux(in_row)
            mask = multid_where(out_rows, output_descri_iter)
            proj[mask, i] = 1.0

        # to get average along j and not sum
        proj /= proj.sum(-1, keepdim=True)

        assert (proj.sum(-1) == 0).sum() == 0

        return proj

    def init_one_chunk(self, input: DescribedTensor, output_descri: Description, i_chunk: int) -> None:
        """ Init the parameters of the model required to compute output_descri from input. """
        self.register_buffer(f'proj_{i_chunk}', self.construct_proj(input, output_descri))

    def get_output_space_dim(self):
        return 2

    def forward_chunk(self, x: torch.tensor, i_chunk: int) -> DescribedTensor:
        """ Performs the average on scale j to make the covariances invariant to scaling. """
        descri = self.descri[i_chunk]

        proj = self.state_dict()[f'proj_{i_chunk}']

        moments = cplx.mm(cplx.from_real(proj), x)

        return DescribedTensor(x=None, descri=descri, y=moments)
