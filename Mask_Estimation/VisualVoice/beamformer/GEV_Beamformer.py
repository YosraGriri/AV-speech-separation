import numpy as np

class GEVBeamformer:
    def __init__(self, gamma=1e-6):
        self.gamma = gamma  # Regularization parameter for conditioning the covariance matrices

    def compute_psd_matrix(self, observation, mask=None, normalize=True):
        """
        Calculate the weighted power spectral density matrix using the observation and mask.
        """
        bins, sensors, frames = observation.shape
        if mask is None:
            mask = np.ones((bins, frames))
        if mask.ndim == 2:
            mask = mask[:, np.newaxis, :]
        psd = np.einsum('...dt,...et->...de', mask * observation, observation.conj())
        if normalize:
            normalization = np.sum(mask, axis=-1, keepdims=True)
            psd /= normalization
        return psd

    def condition_covariance(self, x):
        """
        Condition the covariance matrix to improve its conditioning before inversion.
        """
        scale = self.gamma * np.trace(x) / x.shape[-1]
        scaled_eye = np.eye(x.shape[-1]) * scale
        return (x + scaled_eye) / (1 + self.gamma)

    def get_beamforming_vector(self, target_psd_matrix, noise_psd_matrix):
        """
        Calculate the beamforming vector using the GEV approach.
        """
        conditioned_noise_psd = self.condition_covariance(noise_psd_matrix)
        eigenvals, eigenvecs = np.linalg.eigh(target_psd_matrix, conditioned_noise_psd)
        max_eig_index = np.argmax(eigenvals)
        return eigenvecs[:, max_eig_index]

    def apply_beamforming_vector(self, beamforming_vector, mixed_signal):
        """
        Apply the beamforming vector to the mixed signal.
        """
        return np.einsum('...a,...at->...t', beamforming_vector.conj(), mixed_signal)
