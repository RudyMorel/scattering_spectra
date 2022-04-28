from __future__ import annotations
from typing import *
from collections import Iterator, Iterable
import numpy as np
import torch
import pandas as pd

from utils import get_permutation


class IdxInfo(pd.DataFrame):
    def __init__(self, data: Union[pd.Series, pd.DataFrame, List[Iterable]] = None, columns: List[str] = None):
        """Data provides the rows: series, dataframe or list of rows."""
        super(IdxInfo, self).__init__(data=data)
        if columns is not None and not self.empty:
            self.columns = pd.Index(columns)
        self._index_iter = -1

    def size(self) -> int:
        """The number of idx info."""
        return self.shape[0]

    # def add_col(self, param_name: str, param_value: Iterable) -> None:
    #     """Add a new column."""
    #     assert param_name not in self.columns
    #     self[param_name] = param_value

    def add_row(self, row: Iterable) -> None:
        """Append a row."""
        df = super(IdxInfo, self).copy()
        df.loc[self.size()] = row
        self.__init__(data=df)

    def to_array(self, param_list: Optional[Union[str, List[str]]] = None) -> np.ndarray:
        """Return values stored as a 2d array."""
        if param_list is None:
            param_list = self.columns
        elif not isinstance(param_list, list):
            param_list = [param_list]
        return self[param_list].values

    def where(self, **kwargs: Any) -> np.ndarray:
        """Return the mask of rows satisfying kwargs conditions."""
        masks = [np.ones(self.size(), dtype=bool)]
        for key, value in kwargs.items():
            if isinstance(value, (str, int, float)):
                value = [value]
            masks.append(np.isin(self[key], value))
        return np.logical_and.reduce(masks)

    # todo: replace by Optional by | in python 3.10
    # todo: implement a pivot that will work on pandas index levels
    def reduce(self, mask: Optional[np.ndarray] = None, **kwargs) -> IdxInfo:
        """Return the sub IdxInfo induced by mask or kwargs."""
        mask = self.where(**kwargs) if mask is None else mask
        df = super(IdxInfo, self).copy()
        return IdxInfo(data=df[mask])

    def sort(self, by: Optional[List[str]] = None) -> IdxInfo:
        """Return lexicographically sorted idx info."""
        by = by or list(self.columns)
        return IdxInfo(data=self.sort_values(by=by))

    def drop_duplic(self) -> IdxInfo:
        return IdxInfo(data=self.drop_duplicates())

    @staticmethod
    def cat(*idxinfos: IdxInfo) -> IdxInfo:
        """Concatenates self with idxinfos without duplicates."""
        df_merged = pd.concat(idxinfos).drop_duplicates().reset_index(drop=True)
        return IdxInfo(data=df_merged)

    def tile(self, col_name: List[str], values: Iterable) -> IdxInfo:  # todo: should name it sequence ?
        """Given IdxInfo omega, return the union {n} x omega for n in values."""
        if self.size() == 0:
            raise ValueError("Should tile non zero idx info.")
        dfs = []
        for val in values:
            df = self.copy()
            df[col_name] = val
            dfs.append(df)

        return IdxInfo(data=pd.concat(dfs))

    def iter_tuple(self) -> Iterable[NamedTuple]:
        return self.itertuples(index=False, name='IdxInfo')

    def iter_idx_info(self) -> Iterator:
        for i in range(self.size()):
            yield IdxInfo(self.iloc[[i]])

    def __iter__(self) -> Iterator:
        self._index_iter = -1
        return self

    def __next__(self) -> Union[StopIteration, Iterable]:
        """Iterates over tuples."""
        self._index_iter += 1
        if self._index_iter >= self.size():
            self._index_iter = -1
            raise StopIteration
        else:
            return self.iloc[self._index_iter]

    def __repr__(self) -> str:
        return self.copy().__repr__()

    def __str__(self) -> str:
        return self.copy().__str__()


class ModelOutput:  # todo : should inherit both tensor and idxinfo ?
    def __init__(self, x: Optional[torch.Tensor], y: Optional[torch.Tensor], idx_info: IdxInfo):
        self.x = x
        self.y = y
        self.idx_info = idx_info

    def size(self) -> int:
        return self.idx_info.size()

    #TODO: renamne pivot -> stack
    def select(self, mask: Optional[np.ndarray[bool]] = None, pivot: Optional[str] = None, **kwargs) -> torch.tensor:
        if pivot is None:
            mask = self.idx_info.where(**kwargs) if mask is None else mask
            return self.y[mask, ...]
        out_non_pivot = self.reduce(**kwargs)
        possible_values = np.unique(out_non_pivot.idx_info.to_array(pivot))
        d = OrderedDict({val: [] for val in possible_values})
        for i, val in enumerate(out_non_pivot.idx_info[pivot]):
            d[val].append(i)
        return torch.stack([out_non_pivot.y[val, ...] for val in d.values()])

    def reduce(self, mask: Optional[np.ndarray[bool]] = None, **kwargs) -> ModelOutput:
        mask = self.idx_info.where(**kwargs) if mask is None else mask
        return ModelOutput(self.x, self.y[mask, ...], self.idx_info.reduce(mask))

    def apply(self, h: Callable[[torch.Tensor], torch.Tensor]) -> ModelOutput:
        """Apply an operator h: y -> y."""
        return ModelOutput(x=self.x, y=h(self.y), idx_info=self.idx_info)

    def sort(self, by: Optional[List[str]] = None) -> ModelOutput:
        """Sort lexicogrqphicqlly bqsed on idxinfo."""
        idx_info_sorted = self.idx_info.sort(by=by)
        order = get_permutation(self.idx_info.index.values, idx_info_sorted.index.values)
        return ModelOutput(x=self.x, y=self.y[order, ...], idx_info=idx_info_sorted)

    # def drop_column(self, columns: List[str]) -> ModelOutput:
    #     return ModelOutput(x=self.x, idx_info=self.idx_info.drop(columns=columns), y=self.y).sort()

    @staticmethod
    def cat(*model_outputs: ModelOutput) -> ModelOutput:
        idx_info_with_dup = pd.concat([out.idx_info for out in model_outputs])
        idx_info = IdxInfo(idx_info_with_dup.drop_duplicates().reset_index(drop=True))
        y = torch.cat([out.y for out in model_outputs])
        duplicates = idx_info_with_dup.duplicated().values
        return ModelOutput(x=None, y=y[~duplicates, ...], idx_info=idx_info)

    def mean(self, col: str) -> ModelOutput:
        """Regroup and mean values y by column."""
        values = np.unique(self.idx_info[col])
        y_mean = torch.stack([self.reduce(**{col: val}).y for val in values]).mean(0)
        return ModelOutput(x=self.x, y=y_mean, idx_info=self.reduce(**{col: values[0]}).idx_info)

    def save(self, filepath) -> None:
        torch.save({'x': self.x, 'idx_info': self.idx_info, 'y': self.y}, filepath)

    @staticmethod
    def load(filepath) -> ModelOutput:
        ld = torch.load(filepath)
        return ModelOutput(x=ld['x'], idx_info=ld['idx_info'], y=ld['y'])

    def cpu(self) -> ModelOutput:
        return ModelOutput(None if self.x is None else self.x.cpu(), self.y.cpu(), self.idx_info)

    def cuda(self, device=None) -> ModelOutput:
        return ModelOutput(None if self.x is None else self.x.cuda(device=device), self.y.cuda(device=device), self.idx_info)

    def __repr__(self) -> str:
        return self.idx_info.__repr__()

    def __str__(self) -> str:
        return self.idx_info.__str__()
