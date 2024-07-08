import numpy as np
from scipy.linalg import eigh
import librosa
from scipy.io import wavfile
import mir_eval

def blind_analytic_normalization(vector, noise_psd_matrix, eps=1e-16):
    """
    Reduces distortions in beamformed output.
    :param vector: Beamforming vector with shape (..., sensors)
    :param noise_psd_matrix: Noise PSD matrix with shape (..., sensors, sensors)
    :param eps: Small epsilon value to avoid division by zero
    :return: Scaled Beamforming vector with shape (..., sensors)
    """
    M = vector.shape[-1]  # Number of microphones 
    
 
    psi_v = np.einsum('...ij,...j->...i', noise_psd_matrix, vector)

    vH_psi_psi_v = np.einsum('...i,...i->...', vector.conj(), np.einsum('...ij,...j->...i', noise_psd_matrix, psi_v))
    nominator = np.abs(np.sqrt(vH_psi_psi_v / M))

    vH_psi_v = np.einsum('...i,...i->...', vector.conj(), psi_v)
    denominator = np.abs(vH_psi_v)
    
    normalization = nominator / (denominator + eps)
    return vector * normalization[..., np.newaxis]

def get_power_spectral_density_matrix(observation, mask=None, normalize=True):
    """
    Calculates the weighted power spectral density matrix.
    :param observation: Complex observations with shape (bins, sensors, frames)
    :param mask: Masks with shape (bins, frames) or (bins, 1, frames)
    :return: PSD matrix with shape (bins, sensors, sensors)
    """
    bins, sensors, frames = observation.shape

    if mask is None:
        mask = np.ones((bins, frames))
    if mask.ndim == 2:
        mask = mask[:, np.newaxis, :]

    psd = np.zeros((bins, sensors, sensors), dtype=np.complex64)
    for b in range(bins):
        mask_bin = mask[b, :, np.newaxis]
        spec_bin = observation[b]
        weighted_observation = mask_bin * spec_bin
        psd[b] = np.dot(weighted_observation, observation[b].conj().T)
    
    if normalize:
        normalization = np.sum(mask, axis=-1, keepdims=True)
        psd /= normalization
    
    return psd

def condition_covariance(x, gamma):
    """
    Conditions the covariance matrix to improve numerical stability.
    x: Input covariance matrix
    gamma: Regularization parameter
    returns: Conditioned covariance matrix
    """
    scale = gamma * np.trace(x) / x.shape[-1]
    return (x + np.eye(x.shape[-1]) * scale) / (1 + gamma)


def get_gev_vector(target_psd, noise_psd):
    """
    target_psd: Target PSD matrix with shape (frequency_bins, sensors, sensors)
    noise_psd: Noise PSD matrix with shape (frequency_bins, sensors, sensors)
    returns: Set of beamforming vectors with shape (frequency_bins, sensors)
    """
    num_frequency_bins, num_sensors = target_psd.shape[:2]
    gev_vector = np.empty((num_frequency_bins, num_sensors), dtype=np.complex128)

    for freq_bin in range(num_frequency_bins):
        try:
          
            eigenvalues, eigenvectors = eigh(target_psd[freq_bin], noise_psd[freq_bin]) 
            # Select the eigenvector corresponding to the largest eigenvalue
            gev_vector[freq_bin] = eigenvectors[:, -1]
        except np.linalg.LinAlgError:
            # In case of a numerical error, use a fallback beamforming vector
            gev_vector[freq_bin] = np.ones(num_sensors) / np.trace(noise_psd[freq_bin]) * num_sensors

    return gev_vector


def apply_beamforming_vector(beamforming_vector, mixed_signal):
    """
    Applies the beamforming vector to the mixed signal
    beamforming_vector: Beamforming vector with shape (frequency_bins, sensors)
    mixed_signal: Mixed signal with shape (frequency_bins, sensors, time_frames)
    returns: Beamformed signal with shape (frequency_bins, time_frames)
    """
    frequency_bins, sensors, time_frames = mixed_signal.shape
    beamformed_signal = np.zeros((frequency_bins, time_frames), dtype=np.complex128)
    
    for f in range(frequency_bins):
        for t in range(time_frames):
            beamformed_signal[f, t] = np.dot(beamforming_vector[f].conj(), mixed_signal[f, :, t])
    
    return beamformed_signal

def phase_correction(beamforming_vector):
    """
    to reduce distortions due to phase inconsistencies
    beamforming_vector: Beamforming vector with shape (frequency_bins, sensors)
    returns: Phase corrected beamforming vectors
    
    """
    corrected_vector = beamforming_vector.copy()
    num_frequency_bins, num_sensors = corrected_vector.shape

    for f in range(1, num_frequency_bins):
        # Compute the phase adjustment factor
        phase_adjustment = np.angle(np.sum(corrected_vector[f, :] * corrected_vector[f - 1, :].conj()))
        correction_factor = np.exp(-1j * phase_adjustment)
        
        # Apply the phase correction
        corrected_vector[f, :] *= correction_factor

    return corrected_vector


def gev_wrapper_on_masks(mix, noise_mask=None, target_mask=None, normalization=True):
    """
    Applies GEV beamforming using provided masks.
    mix: Mixed signal
    noise_mask: Mask for noise
    target_mask: Mask for target
    normalization: Whether to apply blind analytic normalization
    returns: Beamformed output
    """
    if noise_mask is None and target_mask is None:
        raise ValueError('At least one mask needs to be present!')

    org_dtype = mix.dtype
    mix = mix.astype(np.complex128)
    mix = mix.T
    if noise_mask is not None:
        noise_mask = noise_mask.T
    if target_mask is not None:
        target_mask = target_mask.T

    target_psd_matrix = get_power_spectral_density_matrix(mix, target_mask, normalize=False)
    noise_psd_matrix = get_power_spectral_density_matrix(mix, noise_mask, normalize=True)
    noise_psd_matrix = condition_covariance(noise_psd_matrix, 1e-3)
    noise_psd_matrix /= np.trace(noise_psd_matrix, axis1=-2, axis2=-1)[..., None, None]
    W_gev = get_gev_vector(target_psd_matrix, noise_psd_matrix)
    W_gev = phase_correction(W_gev)

    if normalization:
        W_gev = blind_analytic_normalization(W_gev, noise_psd_matrix)
        
    output = apply_beamforming_vector(W_gev, mix)
    output = output.astype(org_dtype)
    return output.T


