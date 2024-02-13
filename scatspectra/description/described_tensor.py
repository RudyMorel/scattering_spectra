""" DescribedTensor is the output class of our model. """
from __future__ import annotations
from typing import Dict, Any
from typing import Tuple
from pathlib import Path
import numpy as np
import torch
import pandas as pd

from scatspectra.utils import get_permutation

""" 
Tensor shapes:
- x: input, of shape  (B, N, T)
- y: output, of shape (B, K, T) where K is the number of coefficients
"""


class DescribedTensor:
    """ Output of a scattering model (e.g. scat network, scat spectra) along 
    with the description of the coefficients outputed."""

    def __init__(self,
                 y: torch.Tensor,
                 df: pd.DataFrame,
                 x: torch.Tensor | None = None,
                 config: Dict[str, Any] | None = None):
        self.x = x
        self.y = y
        self.df = df
        self.config = config
        super(DescribedTensor, self).__init__()

    def clone(self) -> DescribedTensor:
        """ Return a clone of self. """
        return DescribedTensor(
            y=self.y.clone(),
            x=None if self.x is None else self.x.clone(),
            df=self.df.copy(),
            config=self.config
        )

    def save(self, filepath: Path) -> None:
        torch.save({
            'x': self.x, 'y': self, 'df': self.df, 'config': self.config
        }, filepath)

    @staticmethod
    def load(filepath: Path) -> DescribedTensor:
        ld = torch.load(filepath)
        return DescribedTensor(
            x=ld['x'], y=ld['y'], df=ld['df'], config=ld['config']
        )

    def eval(self, 
             query_str: str) -> np.ndarray:
        """ Evaluate a query on the description and return the corresponding 
        subset of the tensor. """
        return self.df.eval(query_str).values

    def query(self,
              query_str: str | None = None,
              **kwargs) -> DescribedTensor:
        if query_str is not None:
            mask = self.df.eval(query_str).values
        else:
            masks = [np.ones(self.df.shape[0], dtype=bool)]
            for key, value in kwargs.items():
                if key not in self.df.columns:
                    raise ValueError(f"Column {key} is not in description.")
                if isinstance(value, (str, int, float)):
                    value = [value]
                masks.append(self.df[key].isin(value).values.astype(bool))
            mask = np.logical_and.reduce(masks)
        return DescribedTensor(
            y=self.y[:, mask, :],
            x=self.x,
            df=self.df[mask],
            config=self.config
        )

    def mean_batch(self) -> DescribedTensor:
        """ Average over batch dimension. """
        return DescribedTensor(
            x=self.x,
            y=self.y.mean(dim=0, keepdim=True),
            df=self.df,
            config=self.config
        )

    def sort(self, by: Tuple[str] | None = None) -> DescribedTensor:
        """ Sort lexicographically based on description. """
        df = self.df.reset_index(drop=True)
        df_sorted = df.sort_values(by=by or list(df.columns))
        order = get_permutation(df.index.values, df_sorted.index.values)
        return DescribedTensor(
            x=self.x, y=self.y[:, order, :], df=df_sorted, config=self.config
        )

    def cpu(self) -> DescribedTensor:
        return DescribedTensor(
            x=None if self.x is None else self.x.detach().cpu(),
            y=self.y.detach().cpu(),
            df=self.df,
            config=self.config
        )

    def cuda(self, device: str | None = None) -> DescribedTensor:
        return DescribedTensor(
            x=None if self.x is None else self.x.cuda(device=device),
            y=self.y.cuda(device=device),
            df=self.df,
            config=self.config
        )
