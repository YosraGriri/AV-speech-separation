import numpy as np
from scipy.linalg import eigh, LinAlgError

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
    mask = np.abs(mask)

    psd = np.einsum('...dt,...et->...de', mask * observation, observation.conj())

    #psd = np.einsum('...dt,...et->...de', mask[0, 0] * observation[0,0], mask[0, 1] * observation[0, 1] .conj())
    if normalize:
        normalization = np.sum(mask, axis=-1, keepdims=True)
        psd /= normalization

    for f in range(bins):
        # Check for negative minimum values in the PSD matrix.
        min_val_noise = np.min(psd[f])
        if min_val_noise < 0:
            print(
                f"Bin {f}, Noise PSD - Min Value: {min_val_noise}, "
                f"Condition Number: {np.linalg.cond(psd[f])}")
            #raise ValueError('PSD Matrix cannot have negative values!')
    print('No negative values in PSD! We are good!')

    return psd


def aggregate_psd_matrices(psd_matrices, complex_spectrograms):
    num_channels = len(psd_matrices)
    freq_bins = psd_matrices[0].shape[3]  # Assuming shape is (1, 2, freq_bins, freq_bins)

    # Initialize the multichannel PSD matrix
    multichannel_psd = np.zeros((freq_bins, num_channels, num_channels))

    # Extract the relevant single-channel PSD matrices
    single_channel_psds = [mat[0, 0, :, :] for mat in psd_matrices]

    for i in range(num_channels):
        for j in range(i, num_channels):  # No need to compute twice since the matrix is symmetric
            if i == j:
                # The diagonal elements are the PSDs from the same channel!!
                multichannel_psd[:, i, j] = np.diag(single_channel_psds[i])
            else:
                # The off-diagonal elements should represent cross-channel correlation
                cross_psd = np.mean(complex_spectrograms[i] * np.conj(complex_spectrograms[j]), axis=1)
                cross_psd = cross_psd[:256]
                multichannel_psd[:, i, j] = cross_psd
                multichannel_psd[:, j, i] = cross_psd  # The PSD matrix is Hermitian

    return multichannel_psd


def condition_covariance(x, gamma):
    """
    Stabilizes the covariance matrix by adding a scaled identity matrix.
    :param x: Covariance matrix
    :param gamma: Regularization parameter
    :return: Regularized covariance matrix
    """
    scale = gamma * np.trace(x) / x.shape[-1]
    scaled_eye = np.eye(x.shape[-1]) * scale
    return (x + scaled_eye) / (1 + gamma)

def get_gev_vector(target_psd_matrix, noise_psd_matrix, base_reg_param=1e-6):
    bins, num_channels, _ = noise_psd_matrix.shape
    noise_psd_reg = np.zeros_like(noise_psd_matrix)
    beamforming_vector = np.empty((bins, num_channels))
    for f in range(bins):
        # Check for negative minimum values in noise PSD matrix
        min_val_noise = np.min(noise_psd_matrix[f])
        if min_val_noise < 0:
            print(
                f"Bin {f}, Noise PSD - Min Value: {min_val_noise}, "
                f"Condition Number: {np.linalg.cond(noise_psd_matrix[f])}")

        # Check for negative minimum values in target PSD matrix
        min_val_target = np.min(target_psd_matrix[f])
        if min_val_target < 0:
            print(
                f"Bin {f}, Target PSD - Min Value: {min_val_target}, "
                f"Condition Number: {np.linalg.cond(target_psd_matrix[f])}")

    2/0

    for f in range(bins):
        regularization_success = False
        reg_param = base_reg_param
        while not regularization_success:
            try:
                # Apply regularization
                reg_identity = reg_param * np.eye(num_channels)
                noise_psd_reg[f, :, :] = noise_psd_matrix[f, :, :] + reg_identity

                # Attempt to compute the eigenvalues and eigenvectors
                eigenvals, eigenvecs = eigh(target_psd_matrix[f, :, :], noise_psd_reg[f, :, :])
                beamforming_vector[f, :] = eigenvecs[:, -1]
                regularization_success = True  # Exit loop if successful
            except LinAlgError:
                # Increase the regularization parameter if the matrix is still not positive definite
                reg_param *= 10
                if reg_param > 1e-3:  # Avoid excessively high regularization
                    print(f"Could not to regularize matrix at bin {f} even with high regularization.")
                    break

    return beamforming_vector
class GEVBeamformer:
    def __init__(self, gamma=1e-6):
        self.gamma = gamma

    def compute_psd_matrix(self, observation, mask=None, normalize=True):
        # Use the previously defined function
        return get_power_spectral_density_matrix(observation, mask, normalize)

    def condition_covariance(self, x):
        # Use the previously defined function
        return condition_covariance(x, self.gamma)

    def get_beamforming_vector(self, target_psd_matrix, noise_psd_matrix):
        # First, condition the noise PSD matrix
        conditioned_noise_psd = self.condition_covariance(noise_psd_matrix)
        # Then, get the GEV beamforming vector
        return get_gev_vector(target_psd_matrix, conditioned_noise_psd)

#Beamform it baby!
multichannel_psd_target = np.load('psd_matrices/psd_matrix_target.npy')
multichannel_psd_noise = np.load ('psd_matrices/psd_matrix_noise.npy')
noise_psd_matrix_real = np.real(multichannel_psd_noise)
noise_psd_matrix_target = np.real(multichannel_psd_target)
#gev_beamformer = GEVBeamformer()
#gev_beamformer.get_beamforming_vector(multichannel_psd_target, multichannel_psd_noise)
