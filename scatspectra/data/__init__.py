import pandas as pd
import pkg_resources


def load_data(filename: str, /) -> pd.DataFrame:
    """Loads a dataset from the package data folder.

    Args:
        filename (str): file name

    Returns:
        pd.DataFrame: dataset
    """
    filepath = pkg_resources.resource_filename(__name__, "" + filename)
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    if filepath.endswith(".json"):
        return pd.read_json(filepath)
    return pd.read_pickle(filepath)


snp_data = load_data("snp_WSJ_08_02_2024.pkl")  # SNP data
