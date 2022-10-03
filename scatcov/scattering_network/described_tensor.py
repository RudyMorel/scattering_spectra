""" DescribedTensor is the output class of our model. """
from __future__ import annotations
from typing import *
from collections import Iterator, Iterable
import numpy as np
import torch
import pandas as pd

from scatcov.utils import get_permutation

""" 
Tensor shapes:
- x: input, of shape  (B, N, T)
- y: output, of shape (B, K, T) where K is the number of coefficients
"""


class Description(pd.DataFrame):
    """ The description of an output tensor. It is a pandas dataframe with K rows. Each row i contains the description
    of a coefficients.
    """
    def __init__(self, data: Union[pd.Series, pd.DataFrame, List[Iterable]] = None, columns: List[str] = None):
        """ Data provides the rows: series, dataframe or list of rows. """
        super(Description, self).__init__(data=data)
        if columns is not None and not self.empty:
            self.columns = pd.Index(columns)
        self._index_iter = -1

    def size(self) -> int:
        """ The number of idx info. """
        return self.shape[0]

    # def add_col(self, param_name: str, param_value: Iterable) -> None:
    #     """Add a new column."""
    #     assert param_name not in self.columns
    #     self[param_name] = param_value

    def add_row(self, row: Iterable) -> None:
        """ Append a row. """
        df = super(Description, self).copy()
        df.loc[self.size()] = row
        self.__init__(data=df)

    def to_array(self, param_list: Optional[Union[str, List[str]]] = None) -> np.ndarray:
        """ Return values stored as a 2d array. """
        if param_list is None:
            param_list = self.columns
        elif not isinstance(param_list, list):
            param_list = [param_list]
        return self[param_list].values

    def where(self, **kwargs: Any) -> np.ndarray:
        """ Return the mask of rows satisfying kwargs conditions. """
        masks = [np.ones(self.size(), dtype=bool)]
        for key, value in kwargs.items():
            if key not in self.columns:
                raise ValueError(f"Column {key} is not in description.")
            if isinstance(value, (str, int, float)):
                value = [value]
            masks.append(self[key].isin(value).values.astype(bool))  # cast as bool type because of NaN values
        return np.logical_and.reduce(masks)

    def reduce(self, mask: Optional[np.ndarray] = None, **kwargs) -> Description:
        """ Return the sub Description induced by mask or kwargs. """
        mask = self.where(**kwargs) if mask is None else mask
        df = super(Description, self).copy()
        return Description(data=df[mask])

    def sort(self, by: Optional[List[str]] = None) -> Description:
        """ Return lexicographically sorted idx info. """
        by = by or list(self.columns)
        return Description(data=self.sort_values(by=by))

    def drop_duplic(self) -> Description:
        """ Drop duplicated rows. """
        return Description(data=self.drop_duplicates())

    def drop_col(self, param_name: str) -> Description:
        """ Drop column. """
        if param_name not in self.columns:
            return self
        return Description(data=self.drop(columns=param_name))

    @staticmethod
    def cat(*descriptions: Description) -> Description:
        """ Concatenates self with descriptions without duplicates. """
        df_merged = pd.concat(descriptions).drop_duplicates().reset_index(drop=True)
        return Description(data=df_merged)

    def tile(self, col_name: List[str], values: Iterable) -> Description:
        """ Given Description omega, return the union {n} x omega for n in values. """
        if self.size() == 0:
            raise ValueError("Should tile non zero idx info.")
        dfs = []
        for val in values:
            df = self.copy()
            df[col_name] = val
            dfs.append(df)

        return Description(data=pd.concat(dfs))

    def iter_tuple(self) -> Iterable[NamedTuple]:
        """ Row (tuple) iterator. """
        return self.itertuples(index=False, name='Description')

    # def iter_idx_info(self) -> Iterator:
    #     for i in range(self.size()):
    #         yield Description(self.iloc[[i]])

    def __iter__(self) -> Iterator:
        self._index_iter = -1
        return self

    def __next__(self) -> Union[StopIteration, Iterable]:
        """ Iterates over tuples. """
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


class DescribedTensor:
    """ Contains input tensor x, output tensor y with its description. Each row i in descri is the description of the
     coefficients at place i in tensor y. """
    def __init__(self,
                 x: Optional[torch.Tensor],
                 y: Optional[torch.Tensor],
                 descri: Description):
        self.x = x
        self.y = y
        self.descri = descri

    def size(self) -> int:
        """ The number of coefficients. """
        return self.descri.size()

    def select(self, mask: Optional[np.ndarray[bool]] = None, pivot: Optional[str] = None, **kwargs) -> torch.tensor:
        """ Select tensor based on its description. """
        if pivot is None:
            mask = self.descri.where(**kwargs) if mask is None else mask
            return self.y[:, mask, ...]
        out_non_pivot = self.reduce(**kwargs)
        possible_values = np.unique(out_non_pivot.descri.to_array(pivot))
        d = OrderedDict({val: [] for val in possible_values})
        for i, val in enumerate(out_non_pivot.descri[pivot]):
            d[val].append(i)
        return torch.stack([out_non_pivot.y[:, val, ...] for val in d.values()])

    def reduce(self, mask: Optional[np.ndarray[bool]] = None, b: Optional[int] = None, **kwargs) -> DescribedTensor:
        """ Return a subtensor along with its description. """
        mask = self.descri.where(**kwargs) if mask is None else mask
        reduced_b = self.y if b is None else self.y[b:b+1, ...]
        return DescribedTensor(self.x, reduced_b[:, mask, ...], self.descri.reduce(mask))

    def apply(self, h: Callable[[torch.Tensor], torch.Tensor]) -> DescribedTensor:
        """ Apply an operator h: y -> y. """
        return DescribedTensor(x=self.x, y=h(self.y), descri=self.descri)

    def sort(self, by: Optional[List[str]] = None) -> DescribedTensor:
        """ Sort lexicographically based on description. """
        descri_original = Description(self.descri.reset_index(drop=True))
        descri_sorted = descri_original.sort(by=by)
        order = get_permutation(descri_original.index.values, descri_sorted.index.values)
        return DescribedTensor(x=self.x, y=self.y[:, order, ...], descri=descri_sorted)

    # def drop_column(self, columns: List[str]) -> DescribedTensor:
    #     return DescribedTensor(x=self.x, idx_info=self.idx_info.drop(columns=columns), y=self.y).sort()

    @staticmethod
    def cat(*described_tensors: DescribedTensor) -> DescribedTensor:
        """ Concatenates tensors as well as their description. """
        descri_with_dup = pd.concat([out.descri for out in described_tensors])
        descri = Description(descri_with_dup.drop_duplicates().reset_index(drop=True))
        y = torch.cat([out.y for out in described_tensors], dim=1)
        duplicates = descri_with_dup.duplicated().values
        return DescribedTensor(x=None, y=y[:, ~duplicates, ...], descri=descri)

    @staticmethod
    def cat_batch(*described_tensors: DescribedTensor) -> DescribedTensor:
        """ Concatenates tensors only on their batch dimension. """
        descri = described_tensors[0].descri
        y = torch.cat([out.y for out in described_tensors], dim=0)
        return DescribedTensor(x=None, y=y, descri=descri)

    def mean(self, col: str) -> DescribedTensor:
        """ Regroup and mean values y by column. """
        values = np.unique(self.descri[col])
        y_mean = torch.stack([self.reduce(**{col: val}).y for val in values]).mean(0)
        return DescribedTensor(x=self.x, y=y_mean, descri=self.reduce(**{col: values[0]}).descri)

    def mean_batch(self) -> DescribedTensor:
        """ Regroup and mean values y by column. """
        return DescribedTensor(x=self.x, y=self.y.mean(0, keepdim=True), descri=self.descri)

    def save(self, filepath) -> None:
        torch.save({'x': self.x, 'descri': self.descri, 'y': self.y}, filepath)

    @staticmethod
    def load(filepath) -> DescribedTensor:
        ld = torch.load(filepath)
        return DescribedTensor(x=ld['x'], descri=ld['descri'], y=ld['y'])

    def cpu(self) -> DescribedTensor:
        return DescribedTensor(None if self.x is None else self.x.cpu(), self.y.cpu(), self.descri)

    def cuda(self, device=None) -> DescribedTensor:
        return DescribedTensor(None if self.x is None else self.x.cuda(device=device), self.y.cuda(device=device),
                               self.descri)

    def __repr__(self) -> str:
        return self.descri.__repr__()

    def __str__(self) -> str:
        return self.descri.__str__()
