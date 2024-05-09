import numpy as np
from torch_complex.tensor import ComplexTensor
from torch_complex import functional as FC

import numpy as np


def create_multichannel_psd(psd_matrices):
    """
    Combines individual channel PSD matrices into a multichannel PSD matrix.

    Args:
        psd_matrices (list of np.ndarray): List containing the PSD matrices of each channel.
                                          Each matrix is expected to have a shape of (1, 2, sensors, sensors).

    Returns:
        np.ndarray: Multichannel PSD matrix.
    """
    if not psd_matrices:
        raise ValueError("The list of PSD matrices cannot be empty.")

    # Print the number of PSD matrices and their individual shapes
    print(f"Total number of PSD matrices: {len(psd_matrices)}")
    print(f"Shape of the first PSD matrix: {psd_matrices[0].shape}")

    # Assuming the shape of each PSD matrix is (1, 2, sensors, sensors)
    _, _, sensors, _ = psd_matrices[0].shape
    num_channels = len(psd_matrices) * 2  # Each matrix contains 2 sets of data

    # Initialize the multichannel PSD matrix
    multichannel_psd = np.zeros((sensors, num_channels * sensors, num_channels * sensors), dtype=np.complex64)
    print(f"Initialized Multichannel PSD matrix shape: {multichannel_psd.shape}")

    # Populate the multichannel PSD matrix
    for idx, psd_matrix in enumerate(psd_matrices):
        for j in range(2):  # Loop over the two data sets within each matrix
            row_start = (idx * 2 + j) * sensors
            col_start = row_start
            print(f"Processing PSD matrix {idx} set {j}: Row start = {row_start}, Col start = {col_start}")

            # Check and print the shape of the specific slice being assigned
            print(f"Shape of psd_matrix[0, j]: {psd_matrix[0, j].shape}")

            multichannel_psd[:, row_start:row_start + sensors, col_start:col_start + sensors] = psd_matrix[0, j]

    return multichannel_psd


def compute_psd_matrix(observation, mask=None, normalize=True):
    """
    Calculate the weighted power spectral density matrix using the observation and mask.
    """
    print(f'The observation shape is: {observation[0, 0].shape}')
    print(f'The mask shape is: {mask[0, 0].shape}')
    bins, sensors, frames = observation.shape
    observation = observation[:, :-1, :]

    if mask is None:
        mask = np.ones((bins, frames))
    if mask.ndim == 2:
        mask = mask[:, np.newaxis, :]
    else:
        # Ensure the mask is compatible with the observation's dimensions.
        mask = mask[:bins, :, :frames]

    psd = np.einsum('...dt,...et->...de', mask * observation, observation.conj())

    #psd = np.einsum('...dt,...et->...de', mask[0, 0] * observation[0,0], mask[0, 1] * observation[0, 1] .conj())
    if normalize:
        normalization = np.sum(mask, axis=-1, keepdims=True)
        psd /= normalization
    return psd
def calculate_psd_matrices(spec_mix, mask_prediction_1, mask_prediction_2, opt):
    """
    Calculate PSD matrices for separated audio sources given mixed spectrogram and predicted masks.

    Args:
    spec_mix (np.ndarray): Mixed audio spectrogram.
    mask_prediction_1 (torch.Tensor): Predicted mask for the first audio source.
    mask_prediction_2 (torch.Tensor): Predicted mask for the second audio source.
    opt (argparse.Namespace): Options containing configuration for processing.

    Returns:
    Tuple[np.ndarray, np.ndarray]: PSD matrices for the first and second audio sources.
    """
    # Assuming get_psd_matrix takes a spectrogram and returns its PSD matrix.
    # Convert predictions to numpy if not already done.
    pred_masks_1 = mask_prediction_1.detach().cpu().numpy()
    pred_masks_2 = mask_prediction_2.detach().cpu().numpy()

    # Apply masks to mixed spectrogram to isolate source spectrograms
    source_spec_1 = spec_mix * pred_masks_1
    source_spec_2 = spec_mix * pred_masks_2

    # Calculate PSD matrices
    psd_matrix_1 = get_psd_matrix(source_spec_1)
    psd_matrix_2 = get_psd_matrix(source_spec_2)

    return psd_matrix_1, psd_matrix_2


def get_psd_matrix(spec):
    """
    Compute the Power Spectral Density (PSD) matrix for a given complex spectrogram.

    The PSD matrix is computed as the outer product of the complex spectrogram with its
    conjugate across time frames, and then summed over all frames to provide the average
    PSD matrix.

    Parameters:
    spec (torch.Tensor): A tensor containing the real and imaginary parts of the complex
                         spectrogram with dimensions (Batch, Channel, Frame, Frequency, 2).
                         The last dimension contains real and imaginary parts respectively.

    Returns:
    torch.Tensor: The computed PSD matrix with dimensions (Batch, Frequency, Channel, Channel),
                  representing the average PSD across all frames for each batch and frequency bin.

    Example:
    spec = torch.rand(10, 8, 100, 257, 2)  # A batch of 10, with 8 channels, 100 frames, 257 frequency bins
    psd_matrix = get_psd_matrix(spec)
    print(psd_matrix.shape)  # Expected output: torch.Size([10, 257, 8, 8])
    """
    complex_spec = ComplexTensor(spec[..., 0], spec[..., 1])  # Create a complex tensor from real and imaginary parts
    complex_spec = complex_spec.permute(0, 3, 1, 2)  # Reorder dimensions to (Batch, Frequency, Channel, Frame)
    psd = FC.einsum("...ct,...et->...tce", [complex_spec, complex_spec.conj()])  # Outer product with the conjugate
    psd = psd.sum(dim=-3)  # Sum over frames to get average PSD
    return psd



def get_psd_matrix_numpy(spec):
    """
    Compute the Power Spectral Density (PSD) matrix for a given complex spectrogram in NumPy.

    Parameters:
    spec (np.ndarray): A numpy array containing the complex spectrogram with dimensions
                       (Batch, Channel, Frame, Frequency).

    Returns:
    np.ndarray: The computed PSD matrix with dimensions (Batch, Frequency, Channel, Channel),
                representing the average PSD across all frames for each batch and frequency bin.
    """
    # Assume spec is a complex numpy array with shape (Batch, Channel, Frame, Frequency)
    complex_spec = spec.transpose(0, 3, 1, 2)  # Reorder dimensions for batch processing
    psd = np.einsum('...ct,...et->...tce', complex_spec, np.conj(complex_spec))  # Outer product
    psd = np.sum(psd, axis=-3)  # Sum over frames to get average PSD
    return psd


