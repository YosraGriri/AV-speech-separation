import numpy as np
from utils import calculate_psd_matrices
# Example file paths
masks_dir1 = "../../data/Masks/00039_00388_mic0/00039_mic0_voice0_mask.npy"
masks_dir2 = "../../data/Masks/00039_00388_mic0/00388_mic0_voice1_mask.npy"

# Load masks
mask_prediction_1 = np.load(masks_dir1)
mask_prediction_2 = np.load(masks_dir2)

psd_matrix_1, psd_matrix_2 = calculate_psd _matrices(spec_mix, mask_prediction_1, mask_prediction_2, opt)