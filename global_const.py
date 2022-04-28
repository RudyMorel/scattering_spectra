import os
import torch

from pathlib import Path

# PATHS
CODE_PATH = Path(os.getcwd())
if CODE_PATH.name != 'Library':
    CODE_PATH = CODE_PATH.parents[0]

DATA_PATH = CODE_PATH / 'data'
FINANCE_DATA_PATH = DATA_PATH / 'finance_data'
IMAGE_DATA_PATH = DATA_PATH / 'image_data'
SYNTHETIC_DATA = DATA_PATH / 'synthetic'

OUTPUT_PATH = CODE_PATH / 'output'
FIGURE_PATH = CODE_PATH / 'figures'
OUTPUT_IMAGE_PATH = CODE_PATH / 'output' / 'image_output'
OUTPUT_DICT_PATH = CODE_PATH / 'output' / 'dictionary_output'

PLOT_SAVE_PATH = CODE_PATH / 'Plots' / 'Saved'

# TENSOR
Tensor = torch.DoubleTensor
