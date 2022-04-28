from typing import *
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from networkx.classes.reportviews import NodeView

from utils import transpose, dfs_tree, concat_list
from scattering_network.model_output import IdxInfo, ModelOutput


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

    def external_surjection(self, output_idx_info: IdxInfo) -> IdxInfo:
        """Return IdxInfo that can be computed on output_idx_info."""
        output_idx_info_l = []
        out_columns = None
        for row in output_idx_info.iter_tuple():
            out_info = self.external_surjection_aux(row)
            if out_columns is None:
                out_columns = list(out_info[0]._asdict().keys())
            output_idx_info_row = IdxInfo(data=out_info, columns=out_columns)
            output_idx_info_l.append(output_idx_info_row)

        return IdxInfo.cat(*output_idx_info_l)

    def internal_surjection(self, row: NamedTuple) -> List[NamedTuple]:
        """From output_idx_info return the idxinfos that depend on output_idx_info to be computed."""
        pass

    def init_one_chunk(self, input: ModelOutput, output_idx_info: IdxInfo, i_chunk: int) -> None:
        """Init the parameters of the model required to compute output_idx_info from input_idx_info."""
        pass

    def clear_params(self) -> None:
        pass

    def get_unique(self, col_name):
        """Get the unique values present in idx_info across all chunks."""
        return list(set.union(*[set(chunk[col_name]) for chunk in self.idx_info]))

    def get_output_space_dim(self):
        """Return the size of last dimensions of ModelOutput."""
        pass

    def count_out_channels(self, **kwargs) -> int:
        """Returns the number of moments of moment_type."""
        return sum([idx_info_chunk.reduce(**kwargs).size() for idx_info_chunk in self.idx_info])

    def forward_chunk(self, x: torch.Tensor, i_chunk: int) -> ModelOutput:
        pass

    def forward(self, x: torch.Tensor, i_chunk: Optional[int] = None) -> ModelOutput:
        if i_chunk is not None:
            return self.forward_chunk(x, i_chunk)
        return ModelOutput.cat(*[self.forward_chunk(x, i_chunk) for i_chunk in range(self.nchunks)]).sort()


class ModuleChunk(nn.Module):
    def __init__(self, module_list: List[SubModuleChunk], chunk_method: str, nchunks: int):
        super(ModuleChunk, self).__init__()
        self.module_list = nn.ModuleList(module_list)
        self.nchunks = nchunks

        # self.time_tracker = TimeTracker()

        self.chunk_method = chunk_method
        if chunk_method not in ['quotient_n', 'graph_optim']:
            raise ValueError("Unkown chunk method.")
        if chunk_method == 'graph_optim':
            print("WARNING. graph_optim method is under development.")

        self.chunk_method = chunk_method

        if nchunks > module_list[0].N and chunk_method == 'quotient_n':
            raise ValueError("Current implementation requires N >= nchunks")

        if chunk_method == 'graph_optim':
            # construct idx_infos
            # self.time_tracker.start('idx info')
            self._idx_info = self.construct_idx_info()

            # construct graph of idx_info dependencies
            # self.time_tracker.start('construct graph')
            self.g = self.construct_graph(self.module_list)
            self.nb_edges = self.g.number_of_edges()

            # chunk graph of idx info
            # self.time_tracker.start('graph chunk')
            # self.g_chunks = self.chunk_graph_old(self.g, nchunks)
            self.g_chunks = self.chunk_graph(self.module_list, self.g, nchunks)
            # self.time_tracker.start('compute score')
            score_common_node = self.score_common_nodes(self.g_chunks)
            score_memory = self.score_memory(self.g_chunks)

            # from chuked graphs to chunked idx_info
            # test = IdxInfo(data=[node[1] for node in self.g_chunks[0] if node[0] == f'Module:{0}'])

            self.idx_info_chunked = [[
                IdxInfo(data=[node[1] for node in g_chunk if node[0] == f'Module:{i}']).sort()
                for g_chunk in self.g_chunks]
                for i in range(len(module_list) + 1)
            ]
        else:
            # translate graph chunks into idxinfo chunks
            # self.time_tracker.start('get info chunk')
            self.idx_info_chunked = self.get_chunks(self.module_list, nchunks)
            self.sum_idx_info = sum([sum([chunk.size() for chunk in idx_info]) for idx_info in self.idx_info_chunked])

        # add chunked idx info to modules
        # self.time_tracker.start('idx_info adding')
        for module, chunks in zip(self.module_list, self.idx_info_chunked[1:]):
            module.idx_info = chunks
            module.nchunks = nchunks

        self.m_types = self.module_list[-1].get_unique('m_type')

    def construct_idx_info(self) -> List[IdxInfo]:
        start_idx_info = IdxInfo(data=[[n] for n in range(self.module_list[0].N)], columns=['n'])
        idx_info = [start_idx_info]
        for module in self.module_list:
            input_info = idx_info[-1]
            output_info = module.external_surjection(input_info).sort()
            idx_info.append(output_info)
        return idx_info

    @staticmethod
    def node_name(n, data):
        return f'Module:{n}', data

    def construct_graph(self, module_list: nn.ModuleList) -> nx.DiGraph:
        """Construct a graph whose nodes are the idx_info over the full network."""
        g = nx.DiGraph()

        for i, module in enumerate(module_list):
            in_idx_info = self._idx_info[i]
            out_idx_info = self._idx_info[i + 1]

            # add output nodes
            # self.time_tracker.start('adding nodes')
            new_nodes = [self.node_name(i + 1, row) for row in out_idx_info.iter_tuple()]
            g.add_nodes_from(new_nodes)

            # add forward edges in_info -> out_info
            for in_row in in_idx_info.iter_tuple():
                # self.time_tracker.start('surjection')
                out_row = module.external_surjection_aux(in_row)
                # self.time_tracker.start('new in out edges')
                in_row_name = self.node_name(i, in_row)
                new_edges = [(in_row_name, self.node_name(i + 1, row)) for row in out_row]
                # self.time_tracker.start('adding in out edges')
                g.add_edges_from(new_edges)

            # add internal edges out_info -> out_info
            for out_row1 in out_idx_info.iter_tuple():
                # self.time_tracker.start('surjection')
                out_row2 = module.internal_surjection(out_row1)
                # self.time_tracker.start('new out out edges')
                out_row_name = self.node_name(i + 1, out_row1)
                new_edges = [(out_row_name, self.node_name(i + 1, row)) for row in out_row2]
                # self.time_tracker.start('adding out out edges')
                g.add_edges_from(new_edges)

        return g

    @staticmethod
    def score_common_nodes(gs: List[NodeView]) -> int:
        """Return the sum of nodes over chunked graphs."""
        return sum([len(g_chunk) for g_chunk in gs])

    def score_memory(self, gs: List[NodeView]) -> int:
        """Return the maximum memory usage over chunked graphs."""
        memories = []
        for g_chunk in gs:
            for i, module in enumerate(self.module_list):
                nodes_i = [node[1] for node in g_chunk if node[0] == f'Module:{i + 1}']
                memories.append(len(nodes_i) * module.get_output_space_dim())
        return max(memories)

    def get_chunks(self, module_list: nn.ModuleList, nchunks) -> List[List[IdxInfo]]:
        """Temporary method to obtain idx_info chunks. Only chunk on in channels, and, if necessary on scales."""
        N = module_list[0].N

        if nchunks > N:
            raise ValueError("Should be more in_channels than chunks.")

        n_chunks = np.array_split(np.arange(N), nchunks)
        idx_info_l = []

        if self.chunk_method == 'quotient_n':
            # prepare idx_info for a unique in_channel
            start_idx_info = IdxInfo(data=[[0]], columns=['n'])
            idx_info_canonical = [start_idx_info]
            for module in self.module_list:
                input_info = idx_info_canonical[-1]
                output_info = module.external_surjection(input_info).sort()
                idx_info_canonical.append(output_info)

            # now duplicate same idx_info for other in_channels
            for n_chunk in n_chunks:
                start_idx_info = IdxInfo(data=[[n] for n in n_chunk], columns=['n'])
                idx_info_this_chunk = [start_idx_info]
                for i, module in enumerate(self.module_list):
                    if module.__class__.__name__ in ['Scattering', 'Marginal']:  # scattering
                        idx_info_this_chunk.append(idx_info_canonical[i+1].tile(['n'], n_chunk))
                    else:
                        idx_info_this_chunk.append(
                            idx_info_canonical[i+1].tile(['n1', 'n2'], [(n, n) for n in n_chunk]))

                idx_info_l.append(idx_info_this_chunk)

        return transpose(idx_info_l)  # since the model is separable in N, the chunks should be independent

    def chunk_graph(self, module_list: nn.ModuleList, g: nx.DiGraph, nchunks: int) -> List[NodeView]:
        """Chunk graph into components C such that for any node in C, all its ancestors are in C."""

        def one_try():
            # todo: should include optimization in order to minimize the total number of idx_info

            # split final_idx_info into nchunks equal chunks
            # todo: replace it by graph attribute: Module: i
            final_nodes = [node for node in g if node[0] == f'Module:{len(module_list)}']
            final_section_graph = g.subgraph(final_nodes)
            components = list(nx.connected_components(final_section_graph.to_undirected()))
            # todo: relies on the fact the connected components have roughtly same size, should implement truely uniform
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

        trials = 100
        best_score = 1e34
        best_try = None
        for i in range(trials):
            node_chunks = one_try()
            score = self.score_memory(node_chunks)
            if score < best_score:
                best_score = score
                best_try = node_chunks

        return best_try

    # def init_chunks(self, x: torch.Tensor) -> None:
    #     """Initialize chunk dynamic in each module given its output chunks idx_info_chunked."""
    #     output_idx_info = self.idx_info_chunked[1:]
    #     require_norm = [module.require_normalization for module in self.module_list]
    #     if sum(require_norm) == 0:
    #         last_module_with_norm = 0
    #     else:
    #         last_module_with_norm = len(require_norm) - 1 - require_norm[::-1].index(max(require_norm))
    #
    #     with torch.no_grad():
    #         for i_chunk in range(self.nchunks):
    #
    #             # initially, keep only the in channels required
    #             x_start = torch.stack([x[:, n, :, :] for n in self.idx_info_chunked[0][i_chunk].to_array('n')[:, 0]], 1)
    #
    #             outputs = [ModelOutput(x=None, y=x_start, idx_info=self.idx_info_chunked[0][i_chunk])]
    #
    #             for i_module, (out_info, module) in enumerate(zip(output_idx_info, self.module_list)):
    #                 module.init_one_chunk(outputs[-1], out_info[i_chunk], i_chunk)
    #                 if i_module < last_module_with_norm:
    #                     outputs.append(module.forward_chunk(outputs[-1].y, i_chunk))
    #                 else:
    #                     outputs.append(ModelOutput(x=None, y=None, idx_info=out_info[i_chunk]))

    def init_chunks(self) -> None:
        """Initialize chunk dynamic in each module given its output chunks idx_info_chunked."""
        output_idx_info = self.idx_info_chunked[1:]
        for i_chunk in range(self.nchunks):
            outputs = [ModelOutput(x=None, y=None, idx_info=self.idx_info_chunked[0][i_chunk])]
            for i_module, (out_info, module) in enumerate(zip(output_idx_info, self.module_list)):
                # if self.chunk_method == 'quotient_n' and i_chunk > 0:
                #     # TODO. Horrible efficiency hack
                # else:
                #     module.init_one_chunk(outputs[-1], out_info[i_chunk], i_chunk)
                module.init_one_chunk(outputs[-1], out_info[i_chunk], i_chunk)
                outputs.append(ModelOutput(x=None, y=None, idx_info=out_info[i_chunk]))

    def clear_params(self) -> None:
        for module in self.module_list:
            module.clear_params()

    def count_out_channels(self, **kwargs) -> int:
        # todo: seems broken
        return self.module_list[-1].count_out_channels(**kwargs)

    def forward_chunk(self, y: torch.Tensor, i_chunk: int = None) -> ModelOutput:
        for i, module in enumerate(self.module_list):
            y = module(y, i_chunk)
            if i < len(self.module_list) - 1:
                y = y.y
        return y

    def forward(self, y: torch.Tensor, i_chunk: Optional[int] = None) -> ModelOutput:
        if i_chunk is not None:
            return self.forward_chunk(y, i_chunk)  # no need to sort
        output_l = []
        for i_chunk in range(self.nchunks):
            output_l.append(self.forward_chunk(y, i_chunk))
        return ModelOutput.cat(*output_l).sort()
