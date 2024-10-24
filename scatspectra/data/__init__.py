import numpy as np
import pandas as pd
import pkg_resources


def load_data(filename: str, /) -> pd.DataFrame | np.ndarray:
    """ Loads a dataset from the package data folder.

    :param x: file name
    :return: dataset
    """
    filepath = pkg_resources.resource_filename(__name__, "" + filename)
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    if filepath.endswith(".json"):
        return pd.read_json(filepath)
    if filepath.endswith(".npy"):
        return np.load(filepath)
    return pd.read_pickle(filepath)


snp_data = load_data("snp_WSJ_08_02_2024.pkl")  # SnP data
snp_vix_data = load_data("VIX_SPX_log_return.npy")  # SnP/VIX data  #TODO: should replace the data with .csv with dates
