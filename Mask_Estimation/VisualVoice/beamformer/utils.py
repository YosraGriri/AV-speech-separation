import numpy as np
from torch_complex.tensor import ComplexTensor
from torch_complex import functional as FC



def calculate_psd_matrices(spec_mix, mask_prediction_1, mask_prediction_2, opt):
    """
    Calculate PSD matrices for separated audio sources given mixed spectrogram and predicted masks
    using NumPy arrays.

    Args:
    spec_mix (np.ndarray): Mixed audio spectrogram in NumPy array format.
    mask_prediction_1 (np.ndarray): Predicted mask for the first audio source as a NumPy array.
    mask_prediction_2 (np.ndarray): Predicted mask for the second audio source as a NumPy array.
    opt (argparse.Namespace): Options containing configuration for processing.

    Returns:
    Tuple[np.ndarray, np.ndarray]: PSD matrices for the first and second audio sources as NumPy arrays.
    """
    # Apply masks to mixed spectrogram to isolate source spectrograms
    source_spec_1 = spec_mix * mask_prediction_1
    source_spec_2 = spec_mix * mask_prediction_2

    # Calculate PSD matrices using the NumPy version of the PSD calculation function
    psd_matrix_1 = get_psd_matrix_numpy(source_spec_1)
    psd_matrix_2 = get_psd_matrix_numpy(source_spec_2)

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


