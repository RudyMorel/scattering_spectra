import pandas as pd
import os

file_dir = os.path.dirname(__file__)

s_and_p_data = pd.read_csv(os.path.join(file_dir, 'snp_WSJ_08_02_2024.py'), index_col=0, parse_dates=True)