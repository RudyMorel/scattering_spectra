""" Utils function for manipulating collections. """
from typing import Sequence
from functools import reduce
import numpy as np
import pandas as pd


def list_split(input_list, num_splits):
    """ Split a list into multiple sub-lists. """

    if num_splits > len(input_list):
        raise ValueError("Cannot split a list with more splits than its actual size.")

    # calculate the approximate size of each sublist
    avg_size = len(input_list) // num_splits
    remainder = len(input_list) % num_splits

    # initialize variables
    start = 0
    end = avg_size
    sublists = []

    for i in range(num_splits):
        # adjust sublist size for the remainder
        if i < remainder:
            end += 1

        # create a sublist and add it to the result
        sublist = input_list[start:end]
        sublists.append(sublist)

        # update the start and end indices for the next sublist
        start = end
        end += avg_size

    return sublists
    

def format_args_to_list(*args, n: int | None = None):
    """ Format arguments to lists. """
    n = n or 1
    return [arg if isinstance(arg, list) else [arg] * n for arg in args]


def get_permutation(a: Sequence | np.ndarray, b: Sequence | np.ndarray):
    """Return the permutation s such that a[s[i]] = b[i]"""
    assert set(a) == set(b)

    d = {val: key for key, val in enumerate(a)}
    s = [d[val] for val in b]

    return s


def df_product(*dfs: pd.DataFrame) -> pd.DataFrame:
    for df in dfs:
        df['key'] = 1
    return reduce(lambda l, r: pd.merge(l, r, on='key'), dfs).drop(columns='key')


def df_product_channel_single(df: pd.DataFrame, 
                              N: int, 
                              method: str) -> pd.DataFrame:
    """ Pandas cartesian product {(0,0), ..., (0, Nl-1))} x df """
    if method == "same":
        df_n = pd.DataFrame(np.stack([np.arange(N), np.arange(N)], 1), columns=['nl', 'nr'])
    elif method == "zero_left":
        df_n = pd.DataFrame(np.stack([np.zeros(N, dtype=np.int32), np.arange(N)], 1), columns=['nl', 'nr'])
    elif method == "zero_right":
        df_n = pd.DataFrame(np.stack([np.arange(N), np.zeros(N, dtype=np.int32)], 1), columns=['nl', 'nr'])
    else:
        raise ValueError("Unrecognized channel product method.")

    return df_product(df_n, df)


def df_product_channel_double(df: pd.DataFrame, Nl: int, Nr: int) -> pd.DataFrame:
    """ Pandas cartesian product [0, ..., Nl - 1] x [0, ..., Nr - 1] x df """
    df_nl = pd.DataFrame(np.arange(Nl), columns=['nl'])
    df_nr = pd.DataFrame(np.arange(Nr), columns=['nr'])

    return df_product(df_nl, df_nr, df)