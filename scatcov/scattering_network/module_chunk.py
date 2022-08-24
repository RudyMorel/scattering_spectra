""" Torch modules that can be chunked automatically. """
from typing import *
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn

import scatcov.utils.complex_utils as cplx
from scatcov.utils import transpose
from scatcov.scattering_network.described_tensor import Description, DescribedTensor

class SubModuleChunk(nn.Module):
    """ A module that can be chunked automatically. """
    def __init__(self, init_with_input: bool = False):
        super(SubModuleChunk, self).__init__()

        self.init_with_input = init_with_input

        self.columns = None  # the columns of the output description
        self.descri = None  # List[Description] the output description on each chunk
        self.nchunks = None  # List[Description] the output description on each chunk

    def external_surjection_aux(self, input_descri: NamedTuple) -> List[NamedTuple]:
        """ Return description that can be computed on input_descri. """
        pass

    def external_surjection(self, input_descri: Description) -> Description:
        """ Return Description that can be computed on input_descri. """
        output_descri_l = []
        out_columns = None
        for row in input_descri.iter_tuple():
            out_info = self.external_surjection_aux(row)
            if out_columns is None:
                out_columns = list(out_info[0]._asdict().keys())
            output_descri_row = Description(data=out_info, columns=out_columns)
            output_descri_l.append(output_descri_row)

        return Description.cat(*output_descri_l)

    def internal_surjection(self, output_descri_row: NamedTuple) -> List[NamedTuple]:
        """ Return rows that can be computed on output_descri_row. """
        pass

    def init_one_chunk(self, input: DescribedTensor, output_descri: Description, i_chunk: int) -> None:
        """ Init the parameters of the model required to compute output_descri from input. """
        pass

    def set_descri(self, descri: List[Description], previous_descri: Optional[List[Description]] = None) -> None:
        """ Set output_description of the model. """
        self.descri = descri

    def clear_params(self) -> None:
        pass

    def get_unique(self, col_name):
        """ Get the unique values present in descri across all chunks. """
        return list(set.union(*[set(chunk.loc[:, chunk.columns.isin([col_name])].values.ravel()) for chunk in self.descri]))

    def get_output_space_dim(self):
        """ Return the size of last dimensions of DescribedTensor. """
        pass

    def count_out_channels(self, **kwargs) -> int:
        """ Returns the number of moments satisfying kwargs. """
        return sum([descri_chunk.reduce(**kwargs).size() for descri_chunk in self.descri])

    def forward_chunk(self, x: torch.Tensor, i_chunk: int) -> DescribedTensor:
        """ Forward on one chunk. """
        pass

    def forward(self, x: torch.Tensor, i_chunk: Optional[int] = None) -> DescribedTensor:
        """ Forward on all chunks. """
        if i_chunk is not None:
            return self.forward_chunk(x, i_chunk)
        return DescribedTensor.cat(*[self.forward_chunk(x, i_chunk) for i_chunk in range(self.nchunks)]).sort()


class ModuleChunk(nn.Module):
    """ A Module list that can be chunked and whose forward returns a DescribedTensor. """
    def __init__(self, module_list: List[SubModuleChunk], B: int, N: int, nchunks: int):
        super(ModuleChunk, self).__init__()
        if nchunks > B:
            raise ValueError("Too many chunks for the number of batches.")

        self.N = N
        self.B = B
        self.nchunks = nchunks

        self.module_list = nn.ModuleList(module_list)

        self.batches, self.descri_chunked = self.get_chunks(nchunks)
        self.sum_descri = sum([sum([chunk.size() for chunk in descri]) for descri in self.descri_chunked])

        # add chunked idx info to modules
        for module, pre_chunks, chunks in zip(self.module_list, self.descri_chunked[:-1], self.descri_chunked[1:]):
            module.set_descri(chunks, pre_chunks)
            module.nchunks = nchunks

        self.m_types = self.module_list[-1].get_unique('m_type')

    def construct_description_sequence(self) -> List[Description]:
        """ Return the sequence of descriptions of the tensors outputed by the different modules. """
        start_descri = Description(data=[[n] for n in range(self.N)], columns=['n1'])
        descri = [start_descri]
        for module in self.module_list:
            input_descri = descri[-1]
            output_descri = module.external_surjection(input_descri).sort()
            descri.append(output_descri)
        return descri

    @staticmethod
    def node_name(n, data):
        return f'Module:{n}', data

    @staticmethod
    def score_common_nodes(gs: List) -> int:
        """ Return the sum of nodes over chunked graphs. """
        return sum([len(g_chunk) for g_chunk in gs])

    def score_memory(self, gs: List) -> int:
        """ Return the maximum memory usage over chunked graphs. """
        memories = []
        for g_chunk in gs:
            for i, module in enumerate(self.module_list):
                nodes_i = [node[1] for node in g_chunk if node[0] == f'Module:{i + 1}']
                memories.append(len(nodes_i) * module.get_output_space_dim())
        return max(memories)

    def get_chunks(self, nchunks) -> Tuple[List[np.ndarray], List[List[Description]]]:
        """ Obtain description chunks. The model works on 3 dimensions, batch, channels, scales.
        First try to chunk the batch dimension and then, if necessary the (channels, scales) dimensions.
        """

        if nchunks > self.B:
            raise ValueError("Too much chunked required for current method based on batch only.")

        # batch chunks
        batches = np.array_split(np.arange(self.B), nchunks)

        # (channel, scale) chunks
        start_descri = Description(data=[[n, 0, -1, -1, -1] for n in range(self.N)],
                                   columns=['n1', 'r', 'sc', 'a', 'low'])

        descri_canonical = [start_descri]
        for module in self.module_list:
            input_info = descri_canonical[-1]
            # remove auxillary columns (skip_conn, ..)
            output_info = module.external_surjection(input_info).sort().drop_col('skip_conn')
            descri_canonical.append(output_info)

        descri_l = [descri_canonical] * nchunks

        description_l = transpose(descri_l)
        return batches, description_l

    def init_chunks(self) -> None:
        """ Initialize chunk dynamic in each module given its output chunks descri_chunked. """
        output_descri = self.descri_chunked[1:]
        for i_chunk in range(self.nchunks):
            outputs = [DescribedTensor(x=None, y=None, descri=self.descri_chunked[0][i_chunk])]
            for i_module, (out_descri, module) in enumerate(zip(output_descri, self.module_list)):
                module.init_one_chunk(outputs[-1], out_descri[i_chunk], i_chunk)
                outputs.append(DescribedTensor(x=None, y=None, descri=out_descri[i_chunk]))

    def clear_params(self) -> None:
        for module in self.module_list:
            module.clear_params()

    def count_coefficients(self, **kwargs) -> int:
        """ Return the number of coefficients satisfying kwargs. """
        return self.module_list[-1].count_out_channels(**kwargs)

    def forward_chunk(self, y: torch.Tensor, i_chunk: int = None) -> DescribedTensor:
        """ Forward on one chunk. """
        for i, module in enumerate(self.module_list):
            y = module(y, i_chunk)
            if i < len(self.module_list) - 1:
                y = y.y
        return y

    def forward(self, y: torch.Tensor, i_chunk: Optional[int] = None) -> DescribedTensor:
        """ Forward on all chunk. """
        if i_chunk is not None:
            return self.forward_chunk(y, i_chunk)  # no need to sort
        output_l = []
        for i_chunk, batch in zip(range(self.nchunks), self.batches):
            output_l.append(self.forward_chunk(y[batch, ...], i_chunk))
        return DescribedTensor.cat_batch(*output_l).sort()


class Modulus(SubModuleChunk):
    """ Pointwise modulus. """
    def __init__(self):
        super(Modulus, self).__init__(init_with_input=False)

    def external_surjection_aux(self, input_descri: NamedTuple) -> List[NamedTuple]:
        """ Return description that can be computed on input_descri. """
        return [input_descri]

    def internal_surjection(self, output_descri_row: NamedTuple) -> List[NamedTuple]:
        """ Return rows that can be computed on output_descri_row. """
        return []

    def init_one_chunk(self, input: DescribedTensor, output_descri: Description, i_chunk: int) -> None:
        """ Init the parameters of the model required to compute output_descri from input. """
        pass

    def clear_params(self) -> None:
        pass

    def forward_chunk(self, x: torch.Tensor, i_chunk: int) -> DescribedTensor:
        """ Forward on one chunk. """
        descri = self.descri[i_chunk]

        y = cplx.real(cplx.modulus(x))

        return DescribedTensor(x=None, descri=descri, y=y)


class SkipConnection(SubModuleChunk):
    """ Skip connection. """
    def __init__(self, sub_module: SubModuleChunk):
        super(SkipConnection, self).__init__(sub_module.init_with_input)

        self.sub_module = sub_module

    def external_surjection_aux(self, input_descri: NamedTuple) -> List[NamedTuple]:
        """ Return description that can be computed on input_descri. """
        def add(nt, key, value):
            d = {key: value, **nt._asdict()}
            return namedtuple('Descri', d)(**d)

        out_descri = self.sub_module.external_surjection_aux(input_descri)

        # "skip_connection" column only used to sort the description in the order [x, module(X))].
        return [add(input_descri, 'skip_conn', 0)] + [add(descri, 'skip_conn', 1) for descri in out_descri]

    def internal_surjection(self, output_descri_row: NamedTuple) -> List[NamedTuple]:
        """ Return rows that can be computed on output_descri_row. """
        return self.sub_module.internal_surjection(output_descri_row)

    def init_one_chunk(self, input: DescribedTensor, output_descri: Description, i_chunk: int) -> None:
        """ Init the parameters of the model required to compute output_descri from input. """
        sub_module_out_descri = Description(output_descri.iloc[input.descri.size():])
        self.sub_module.init_one_chunk(input, sub_module_out_descri, i_chunk)

    def set_descri(self, descri: List[Description], previous_descri: Optional[List[Description]] = None) -> None:
        """ Set output_description of the model. """
        self.descri = descri
        sub_module_descri = [Description(descri.iloc[pre_descri.size():])
                             for (pre_descri, descri) in zip(previous_descri, descri)]
        self.sub_module.set_descri(sub_module_descri)

    def clear_params(self) -> None:
        self.sub_module.clear_params()

    def forward_chunk(self, x: torch.Tensor, i_chunk: int) -> DescribedTensor:
        """ Forward on one chunk. """
        descri = self.descri[i_chunk]

        y = self.sub_module(x, i_chunk).y

        return DescribedTensor(x=None, descri=descri, y=torch.cat([x, y], dim=1))
