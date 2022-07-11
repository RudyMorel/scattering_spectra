from typing import *
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn

import utils.complex_utils as cplx
from utils import transpose
from scattering_network.described_tensor import Description, DescribedTensor


class SubModuleChunk(nn.Module):
    def __init__(self, require_normalization: bool = False):
        super(SubModuleChunk, self).__init__()

        self.require_normalization = require_normalization

        self.columns = None  # the columns of the outputIdxInfo
        self.idx_info = None  # List[IdxInfo] the output idx_info on each chunk
        self.nchunks = None  # List[IdxInfo] the output idx_info on each chunk

    def external_surjection_aux(self, input_idx_info: NamedTuple) -> List[NamedTuple]:
        """Return IdxInfo that can be computed on input_idx_info."""
        pass

    def external_surjection(self, output_idx_info: Description) -> Description:
        """Return IdxInfo that can be computed on output_idx_info."""
        output_idx_info_l = []
        out_columns = None
        for row in output_idx_info.iter_tuple():
            out_info = self.external_surjection_aux(row)
            if out_columns is None:
                out_columns = list(out_info[0]._asdict().keys())
            output_idx_info_row = Description(data=out_info, columns=out_columns)
            output_idx_info_l.append(output_idx_info_row)

        return Description.cat(*output_idx_info_l)

    def internal_surjection(self, row: NamedTuple) -> List[NamedTuple]:
        """From output_idx_info return the idxinfos that depend on output_idx_info to be computed."""
        pass

    def init_one_chunk(self, input: DescribedTensor, output_idx_info: Description, i_chunk: int) -> None:
        """Init the parameters of the model required to compute output_idx_info from input_idx_info."""
        pass

    def set_idx_info(self, idx_info: List[Description], previous_idx_info: Optional[List[Description]] = None) -> None:
        """Set idx_info of the model."""
        self.idx_info = idx_info

    def clear_params(self) -> None:
        pass

    def get_unique(self, col_name):
        """Get the unique values present in idx_info across all chunks."""
        return list(set.union(*[set(chunk.loc[:, chunk.columns.isin([col_name])].values.ravel()) for chunk in self.idx_info]))

    def get_output_space_dim(self):
        """Return the size of last dimensions of ModelOutput."""
        pass

    def count_out_channels(self, **kwargs) -> int:
        """Returns the number of moments of moment_type."""
        return sum([idx_info_chunk.reduce(**kwargs).size() for idx_info_chunk in self.idx_info])

    def forward_chunk(self, x: torch.Tensor, i_chunk: int) -> DescribedTensor:
        pass

    def forward(self, x: torch.Tensor, i_chunk: Optional[int] = None) -> DescribedTensor:
        if i_chunk is not None:
            return self.forward_chunk(x, i_chunk)
        return DescribedTensor.cat(*[self.forward_chunk(x, i_chunk) for i_chunk in range(self.nchunks)]).sort()


class Modulus(SubModuleChunk):
    def __init__(self):
        super(Modulus, self).__init__(require_normalization=False)

    def external_surjection_aux(self, input_idx_info: NamedTuple) -> List[NamedTuple]:
        """Return IdxInfo that can be computed on input_idx_info."""
        return [input_idx_info]

    def internal_surjection(self, row: NamedTuple) -> List[NamedTuple]:
        """From output_idx_info return the idxinfos that depend on output_idx_info to be computed."""
        return []

    def init_one_chunk(self, input: DescribedTensor, output_idx_info: Description, i_chunk: int) -> None:
        """Init the parameters of the model required to compute output_idx_info from input_idx_info."""
        pass

    def clear_params(self) -> None:
        pass

    def forward_chunk(self, x: torch.Tensor, i_chunk: int) -> DescribedTensor:
        idx_info = self.idx_info[i_chunk]

        y = cplx.real(cplx.modulus(x))

        return DescribedTensor(x=None, idx_info=idx_info, y=y)


class SkipConnection(SubModuleChunk):
    def __init__(self, sub_module: SubModuleChunk):
        super(SkipConnection, self).__init__(sub_module.require_normalization)

        self.sub_module = sub_module

    def external_surjection_aux(self, input_idx_info: NamedTuple) -> List[NamedTuple]:
        """Return IdxInfo that can be computed on input_idx_info."""

        def add(nt, key, value):
            d = {key: value, **nt._asdict()}
            return namedtuple('IdxInfo', d)(**d)

        out_idx_info = self.sub_module.external_surjection_aux(input_idx_info)

        # "skip_connection" column only used to sort the description in the order [x, module(X))].
        return [add(input_idx_info, 'skip_conn', 0)] + [add(idx_info, 'skip_conn', 1) for idx_info in out_idx_info]

    def internal_surjection(self, row: NamedTuple) -> List[NamedTuple]:
        """From output_idx_info return the idxinfos that depend on output_idx_info to be computed."""
        return self.sub_module.internal_surjection(row)

    def init_one_chunk(self, input: DescribedTensor, output_idx_info: Description, i_chunk: int) -> None:
        """Init the parameters of the model required to compute output_idx_info from input_idx_info."""
        sub_module_out_idx_info = Description(output_idx_info.iloc[input.idx_info.size():])
        self.sub_module.init_one_chunk(input, sub_module_out_idx_info, i_chunk)

    def set_idx_info(self, idx_info: List[Description], previous_idx_info: Optional[List[Description]] = None) -> None:
        """Set idx_info of the model."""
        self.idx_info = idx_info
        sub_module_idx_info = [Description(idx_info.iloc[pre_idx_info.size():])
                               for (pre_idx_info, idx_info) in zip(previous_idx_info, idx_info)]
        self.sub_module.set_idx_info(sub_module_idx_info)

    def clear_params(self) -> None:
        self.sub_module.clear_params()

    def forward_chunk(self, x: torch.Tensor, i_chunk: int) -> DescribedTensor:
        idx_info = self.idx_info[i_chunk]

        y = self.sub_module(x, i_chunk).y

        return DescribedTensor(x=None, idx_info=idx_info, y=torch.cat([x, y]))


class ModuleChunk(nn.Module):
    def __init__(self, module_list: List[SubModuleChunk], chunk_method: str, N: int, nchunks: int):
        super(ModuleChunk, self).__init__()
        self.module_list = nn.ModuleList(module_list)

        self.chunk_method = chunk_method
        if chunk_method not in ['quotient_n', 'graph_optim']:
            raise ValueError("Unkown chunk method.")
        if chunk_method == 'graph_optim':
            print("WARNING. graph_optim method is under development.")

        self.N = N
        self.nchunks = nchunks
        self.chunk_method = chunk_method

        if nchunks > self.N and chunk_method == 'quotient_n':
            raise ValueError("Current implementation requires N >= nchunks")
        if chunk_method == 'graph_optim':
            raise ValueError("Chunk method graph_optim not yet supported.")

        self.idx_info_chunked = self.get_chunks(nchunks)
        self.sum_idx_info = sum([sum([chunk.size() for chunk in idx_info]) for idx_info in self.idx_info_chunked])

        # add chunked idx info to modules
        for module, pre_chunks, chunks in zip(self.module_list, self.idx_info_chunked[:-1], self.idx_info_chunked[1:]):
            module.set_idx_info(chunks, pre_chunks)
            module.nchunks = nchunks

        self.m_types = self.module_list[-1].get_unique('m_type')

    def construct_idx_info(self) -> List[Description]:
        start_idx_info = Description(data=[[n] for n in range(self.N)], columns=['n1'])
        idx_info = [start_idx_info]
        for module in self.module_list:
            input_info = idx_info[-1]
            output_info = module.external_surjection(input_info).sort()
            idx_info.append(output_info)
        return idx_info

    @staticmethod
    def node_name(n, data):
        return f'Module:{n}', data

    @staticmethod
    def score_common_nodes(gs: List) -> int:
        """Return the sum of nodes over chunked graphs."""
        return sum([len(g_chunk) for g_chunk in gs])

    def score_memory(self, gs: List) -> int:
        """Return the maximum memory usage over chunked graphs."""
        memories = []
        for g_chunk in gs:
            for i, module in enumerate(self.module_list):
                nodes_i = [node[1] for node in g_chunk if node[0] == f'Module:{i + 1}']
                memories.append(len(nodes_i) * module.get_output_space_dim())
        return max(memories)

    def get_chunks(self, nchunks) -> List[List[Description]]:
        """Temporary method to obtain idx_info chunks. Only chunk on in channels, and, if necessary on scales."""
        if nchunks > self.N:
            raise ValueError("Should be more in_channels than chunks.")

        n_chunks = np.array_split(np.arange(self.N), nchunks)
        idx_info_l = []

        if self.chunk_method == 'quotient_n':
            # prepare idx_info for a unique in_channel
            start_idx_info = Description(data=[[0, 0, -1, -1, -1]], columns=['n1', 'r', 'sc', 'a', 'low'])
            idx_info_canonical = [start_idx_info]
            for module in self.module_list:
                input_info = idx_info_canonical[-1]
                # remove auxillary columns (skip_conn, ..)
                output_info = module.external_surjection(input_info).sort().drop_col('skip_conn')
                idx_info_canonical.append(output_info)

            module_names = [module.__class__.__name__ for module in self.module_list]
            start_idx_info = Description(data=[[n, 0, -1, -1, -1] for n in range(self.N)],
                                         columns=['n1', 'r', 'sc', 'a', 'low'])
            idx_info_l = [
                [start_idx_info] +
                [
                    idx_info.tile(['n1', 'n1p'], [(n, n) for n in n_chunk])
                    if name in ['Cov', 'CovStat'] else
                    idx_info.tile(['n1'], n_chunk)
                    for (idx_info, name) in zip(idx_info_canonical[1:], module_names)
                ]
                for n_chunk in n_chunks
            ]

        description_l = transpose(idx_info_l)
        return description_l  # since the model is separable in N, the chunks should be independent

    def init_chunks(self) -> None:
        """Initialize chunk dynamic in each module given its output chunks idx_info_chunked."""
        output_idx_info = self.idx_info_chunked[1:]
        for i_chunk in range(self.nchunks):
            outputs = [DescribedTensor(x=None, y=None, idx_info=self.idx_info_chunked[0][i_chunk])]
            for i_module, (out_info, module) in enumerate(zip(output_idx_info, self.module_list)):
                # if self.chunk_method == 'quotient_n' and i_chunk > 0:
                #     # TODO. Horrible efficiency hack
                # else:
                #     module.init_one_chunk(outputs[-1], out_info[i_chunk], i_chunk)
                module.init_one_chunk(outputs[-1], out_info[i_chunk], i_chunk)
                outputs.append(DescribedTensor(x=None, y=None, idx_info=out_info[i_chunk]))

    def clear_params(self) -> None:
        for module in self.module_list:
            module.clear_params()

    def count_out_channels(self, **kwargs) -> int:
        # todo: seems broken
        return self.module_list[-1].count_out_channels(**kwargs)

    def forward_chunk(self, y: torch.Tensor, i_chunk: int = None) -> DescribedTensor:
        for i, module in enumerate(self.module_list):
            y = module(y, i_chunk)
            if i < len(self.module_list) - 1:
                y = y.y
        return y

    def forward(self, y: torch.Tensor, i_chunk: Optional[int] = None) -> DescribedTensor:
        if i_chunk is not None:
            return self.forward_chunk(y, i_chunk)  # no need to sort
        output_l = []
        for i_chunk in range(self.nchunks):
            output_l.append(self.forward_chunk(y, i_chunk))
        return DescribedTensor.cat(*output_l).sort()
