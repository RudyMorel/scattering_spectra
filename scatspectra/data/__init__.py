import pandas as pd
import os

file_dir = os.path.dirname(__file__)

s_and_p_data = pd.read_csv(os.path.join(file_dir, 'snp_WSJ_08_02_2024.py'), index_col=0, parse_dates=True)

import pandas as pd
import pkg_resources

# Fonction pour charger les données
def load_data(filename):
    filepath = pkg_resources.resource_filename(__name__, 'data/' + filename)
    return pd.read_csv(filepath)

# Variables pour accéder aux données
data_snp = load_data('snp_WSJ_08_02_2024.csv')
