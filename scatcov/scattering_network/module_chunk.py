""" Torch modules that can be chunked automatically. """
from typing import *
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from networkx.classes.reportviews import NodeView

import scatcov.utils.complex_utils as cplx
from scatcov.utils import concat_list, multid_where, dfs_tree
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
        self.N = N
        self.B = B
        self.nchunks = nchunks

        self.module_list = nn.ModuleList(module_list)

        # construct descriptions
        self._descriptions = self.construct_description_sequence()

        # construct graph of descri dependencies between one module to another
        self.g = self.construct_graph(self.module_list)
        self.nb_edges = self.g.number_of_edges()

        # chunk the model, first along input batches then along scales and input channels
        self.batches_per_chunk, self.nchunk_per_batch = self.get_batches()
        self.descri_chunked = self.get_chunks()

        # add chunked idx info to modules
        for module, pre_chunks, chunks in zip(self.module_list, self.descri_chunked[:-1], self.descri_chunked[1:]):
            module.set_descri(chunks, pre_chunks)
            module.nchunks = nchunks

        self.m_types = self.module_list[-1].get_unique('m_type')

    def construct_description_sequence(self) -> List[Description]:
        """ Return the sequence of descriptions of the tensors outputed by the different modules. """
        start_descri = Description(data=[[n, 0, -1, -1, -1] for n in range(self.N)],
                                   columns=['n1', 'r', 'sc', 'a', 'low'])
        descri = [start_descri]
        for module in self.module_list:
            input_descri = descri[-1]
            output_descri = module.external_surjection(input_descri).sort().drop_col('skip_conn')
            descri.append(output_descri)
        return descri

    @staticmethod
    def node_name(n, data):
        return f'Module:{n}', data

    def construct_graph(self, module_list: nn.ModuleList) -> nx.DiGraph:
        """ Construct a graph whose nodes are the description over the full network. """
        g = nx.DiGraph()

        for i, module in enumerate(module_list):
            in_descri = self._descriptions[i]
            out_descri = self._descriptions[i + 1]

            # add output nodes
            new_nodes = [self.node_name(i+1, row) for row in out_descri.iter_tuple()]
            g.add_nodes_from(new_nodes)

            def remove_skipconn(row):
                d = {key: value for (key, value) in row._asdict().items() if key != 'skip_conn'}
                return namedtuple('Descri', d)(**d)

            # add forward edges in_info -> out_info
            for in_row in in_descri.iter_tuple():
                out_row = module.external_surjection_aux(in_row)
                in_row_name = self.node_name(i, in_row)
                new_edges = [(in_row_name, self.node_name(i+1, remove_skipconn(row))) for row in out_row]
                g.add_edges_from(new_edges)

            # add internal edges out_info -> out_info
            for out_row1 in out_descri.iter_tuple():
                out_row2 = module.internal_surjection(out_row1)
                out_row_name = self.node_name(i+1, out_row1)
                new_edges = [(out_row_name, self.node_name(i+1, remove_skipconn(row))) for row in out_row2]
                g.add_edges_from(new_edges)

        return g

    @staticmethod
    def score_common_nodes(gs: List[NodeView]) -> int:
        """ Return the sum of nodes over chunked graphs. """
        return sum([len(g_chunk) for g_chunk in gs])

    def score_memory(self, gs: List[NodeView]) -> int:
        """ Return the maximum memory usage over chunked graphs. """
        memories = []
        for g_chunk in gs:
            for i, module in enumerate(self.module_list):
                nodes_i = [node[1] for node in g_chunk if node[0] == f'Module:{i+1}']
                memories.append(len(nodes_i) * module.get_output_space_dim())
        return max(memories)

    def get_batches(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """ Do the partition of batches into chunks. """
        if self.nchunks <= self.B:
            # batch chunks
            batches = np.array_split(np.arange(self.B), self.nchunks)
            nchunks_per_batch = np.ones(self.nchunks, dtype=np.int32)
        else:
            k, n = self.nchunks // self.B + 1, self.nchunks % self.B
            nchunks_per_batch = (k,) * n + (k-1,) * (self.B-n)
            batches = [np.array([b]) for b, chunk in enumerate(nchunks_per_batch) for _ in range(chunk)]

        return batches, nchunks_per_batch

    def get_chunks(self) -> List[List[Description]]:
        """ Obtain description chunks. The model works on 3 dimensions, batch, channels, scales.
        First try to chunk the batch dimension and then, if necessary the (channels, scales) dimensions.
        """
        descri_l = []
        for n_chunk_this_batch in self.nchunk_per_batch:
            # chunk graph of idx info
            g_chunks = self.chunk_graph(self.module_list, self.g, n_chunk_this_batch)

            descri_chunked = [[
                Description(data=[node[1] for node in g_chunk if node[0] == f'Module:{i}']).sort()
                for g_chunk in g_chunks]
                for i in range(len(self.module_list)+1)
            ]
            descri_l.append(descri_chunked)

        descri_l = [concat_list([descri_l[b][i_module] for b in range(len(descri_l))])
                    for i_module in range(len(self.module_list)+1)]

        return descri_l

    def chunk_graph(self, module_list: nn.ModuleList, g: nx.DiGraph, nchunks: int) -> List[NodeView]:
        """ Chunk graph into components C such that for any node in C, all its ancestors are in C. """

        def one_try():
            # split final_descri into nchunks equal chunks
            final_nodes = [node for node in g if node[0] == f'Module:{len(module_list)}']
            final_section_graph = g.subgraph(final_nodes)
            components = list(nx.connected_components(final_section_graph.to_undirected()))
            size = len(components)
            if size < nchunks:
                raise ValueError("More chunks than the number of out components in the graph.")
            indices = np.array_split(np.random.permutation(size), nchunks)
            final_chunks = [concat_list([components[i] for i in idx]) for idx in indices]

            # split full graph g accordingly
            node_chunks = []
            for chunk_nodes in final_chunks:
                node_chunks.append(dfs_tree(g.reverse(), source=chunk_nodes).nodes)

            return node_chunks

        trials = 1
        best_score = 1e34
        best_try = None
        for i in range(trials):
            node_chunks = one_try()
            # score = self.score_memory(node_chunks)
            score = 0
            if score < best_score:
                best_score = score
                best_try = node_chunks

        return best_try

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

        # chunked forward
        output_l = []
        for i_chunk, batch in enumerate(self.batches_per_chunk):
            output_l.append(self.forward_chunk(y[batch, ...], i_chunk))

        # in that case chunks consisted in splitting the batch dimension
        if self.nchunks <= self.B:
            return DescribedTensor.cat_batch(*output_l)

        # in that case chunks consisted in splitting both batch dimension and the model itself
        batch_cuts = np.cumsum(np.insert(self.nchunk_per_batch, 0, 0))
        output_per_batch = [DescribedTensor.cat(*output_l[a:b]).sort()
                            for (a, b) in zip(batch_cuts[:-1], batch_cuts[1:])]

        return DescribedTensor.cat_batch(*output_per_batch)


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

        # params
        self.masks = []

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
        where_is_skipped = multid_where(output_descri, input.descri)
        idx_true_output = [i for i, is_skipped in enumerate(where_is_skipped) if is_skipped is None]

        positions = multid_where(input.descri, output_descri)
        mask_x = [(pos is not None) for pos in positions]
        mask_skipped = [pos for pos in positions if pos is not None]
        mask_unskipped = [pos for pos in range(output_descri.size()) if pos not in mask_skipped]
        self.masks.append([mask_x, mask_skipped, mask_unskipped])

        sub_module_out_descri = Description(output_descri.iloc[idx_true_output])
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
        mask_x, mask_skipped, mask_unskipped = self.masks[i_chunk]

        y_skip = x.new_zeros(x.shape[0], descri.size(), x.shape[2], 2)
        y_skip[:, mask_skipped, ...] = x[:, mask_x, ...]

        if len(mask_unskipped) > 0:
            y = self.sub_module(x, i_chunk).y
            y_skip[:, mask_unskipped, ...] = y

        return DescribedTensor(x=None, descri=descri, y=y_skip)
