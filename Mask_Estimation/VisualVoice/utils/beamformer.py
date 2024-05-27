import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, LinAlgError
import librosa
import mir_eval

def getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt):
    reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0)), axis=0)
    estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, False)
    return np.mean(sdr), np.mean(sir), np.mean(sar)
def compute_tdoa(r_u, r_v, theta, fs, c):
    r_u = np.array(r_u)
    r_v = np.array(r_v)
    theta = np.array(theta)
    return fs / c * np.dot((r_u - r_v), theta)


def steering_vector(tdoa, f, N):
    return np.exp(1j * 2 * np.pi * f * tdoa / N)


def gain_function(delta_tdoa, alpha, beta):
    return np.exp(-alpha * (delta_tdoa - beta)) / (1 + np.exp(-alpha * (delta_tdoa - beta)))



def get_irms(stft_clean, stft_noise):
    mag_clean = np.abs(stft_clean) ** 2
    mag_noise = np.abs(stft_noise) ** 2
    irm_speech = mag_clean / (mag_clean + mag_noise)
    print(irm_speech.shape)
    irm_noise = mag_noise / (mag_clean + mag_noise)
    return irm_speech[:, 0, :], irm_noise[:, 0, :]
def cirm(y_stft, s_stft, r_u, r_v, theta_t, theta_i, fs, c, alpha, beta, K=10, C=0.1, flat=True):
    tdoa_uv = compute_tdoa(r_u, r_v, theta_t, fs, c)
    delta_tdoa_uv = compute_tdoa(r_u, r_v, theta_t - theta_i, fs, c)

    f = np.arange(y_stft.shape[0])
    N = y_stft.shape[1]

    A_uv = steering_vector(tdoa_uv, f[:, None], N)
    A_uv = A_uv[:, :, None]
    print(f'Shape of A_uv, steering vector is: {A_uv.shape}')
    Y_uv = A_uv * y_stft * np.conj(y_stft)

    G_uv = gain_function(delta_tdoa_uv, alpha, beta)

    S_u = np.abs(s_stft)
    I_u = np.abs(y_stft - s_stft)
    B_u = np.zeros_like(S_u)

    M_uv = (S_u ** 2 + G_uv * I_u ** 2) / (S_u ** 2 + I_u ** 2 + B_u ** 2)

    if flat:
        return M_uv
    else:
        return K * ((1 - np.exp(-C * M_uv)) / (1 + np.exp(-C * M_uv)))

def get_power_spectral_density_matrix(observation, mask=None, normalize=True):
    """
    Calculates the weighted power spectral density matrix.

    This does not yet work with more than one target mask.

    :param observation: Complex observations with shape (bins, sensors, frames)
    :param mask: Masks with shape (bins, frames) or (bins, 1, frames)
    :return: PSD matrix with shape (bins, sensors, sensors)
    """
    bins, sensors, frames = observation.shape
    print(observation.shape)
    # print(mask.shape)

    if mask is None:
        mask = np.ones((bins, frames))
    if mask.ndim == 2:
        mask = mask[:, np.newaxis, :]
        # mask = mask[:,:,  np.newaxis]
        print(f'Mask shape is: {mask.shape}')
    print(f'Spectrogram shape is: {observation.shape}')
    psd = np.zeros((bins, sensors, sensors), dtype=np.complex64)

    for b in range(bins):
        print(bins)
        mask_bin = mask[b, :, np.newaxis]
        #print(mask_bin.shape)
        spec_bin = observation[b]
        #print(spec_bin.shape)
        weighted_observation = mask_bin * spec_bin
        psd[b] = np.dot(weighted_observation, observation[b].conj().T)
    print(f'PSD shape is: {psd.shape}')
    if normalize:
        normalization = np.sum(mask, axis=-1, keepdims=True)
        psd /= normalization
    return psd


def condition_covariance(x, gamma):

    scale = gamma * np.trace(x) / x.shape[-1]
    scaled_eye = np.eye(x.shape[-1]) * scale
    return (x + scaled_eye) / (1 + gamma)


def get_gev_vector(target_psd_matrix, noise_psd_matrix):
    """
    Returns the GEV beamforming vector.
    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """
    bins, sensors, _ = target_psd_matrix.shape
    beamforming_vector = np.empty((bins, sensors), dtype=np.complex128)
    for f in range(bins):
        try:
            eigenvals, eigenvecs = eigh(target_psd_matrix[f, :, :],
                                        noise_psd_matrix[f, :, :])
            beamforming_vector[f, :] = eigenvecs[:, -1]
        except np.linalg.LinAlgError:
            print('LinAlg error for frequency {}'.format(f))
            beamforming_vector[f, :] = (
                    np.ones((sensors,)) / np.trace(noise_psd_matrix[f]) * sensors
            )
    return beamforming_vector
def get_irms(stft_clean, stft_noise):
    mag_clean = np.abs(stft_clean) ** 2
    mag_noise = np.abs(stft_noise) ** 2
    irm_speech = mag_clean / (mag_clean + mag_noise+1e-16)
    print(irm_speech.shape)
    irm_noise = mag_noise / (mag_clean + mag_noise+1e-16)
    return irm_speech[:, :], irm_noise[:, :]

def istft_reconstruction_from_complex(real, imag, hop_length=160, win_length=400, length=65535):
    spec = real + 1j*imag
    wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length, length=length)
    return np.clip(wav, -1., 1.)
def gev_wrapper_on_masks(mix, noise_mask=None, target_mask=None, normalization=False, gamma=1e-1):
    org_dtype = mix.dtype
    mix = mix.astype(np.complex128)
    mix = mix.T

    # Calculate PSD matrices
    target_psd_matrix = get_power_spectral_density_matrix(mix, target_mask, normalize=True)
    noise_psd_matrix = get_power_spectral_density_matrix(mix, noise_mask, normalize=True)

    # Debug: Print or plot PSD matrices
    #plot_psd_matrix_channels(target_psd_matrix, 0, 0, "Target PSD Matrix Sensor 0")
    #plot_psd_matrix_channels(noise_psd_matrix, 0, 0, "Noise PSD Matrix Sensor 0")

    # Debug: Print or plot PSD matrices
    #debug_psd(target_psd_matrix, "Target PSD Matrix")
    #debug_psd(noise_psd_matrix, "Noise PSD Matrix")

    # Condition the noise PSD matrix
    print(f'noise_psd_matrix shape is: {noise_psd_matrix.shape}')
    noise_psd_matrix = condition_covariance(noise_psd_matrix, gamma)
    noise_psd_matrix /= np.trace(noise_psd_matrix, axis1=-2, axis2=-1)[..., None, None]
    print(f'noise_psd_matrix after covariance shape is: {noise_psd_matrix.shape}')
    # Debug: Print or plot conditioned noise PSD matrix
    #debug_psd(noise_psd_matrix, "Conditioned Noise PSD Matrix")

    # Get GEV beamforming vector
    W_gev = get_gev_vector(target_psd_matrix, noise_psd_matrix)
    W_gev = phase_correction(W_gev)

    #if normalization:
        #W_gev = blind_analytic_normalization(W_gev, noise_psd_matrix)

    # Debug: Print or plot GEV beamforming vector
    print(f"GEV beamforming vector shape: {W_gev.shape}")

    # Applyingbeamforming
    print(f"mix vector shape: {mix.shape}")
    output = apply_beamforming_vector(W_gev, mix)
    output = output.astype(org_dtype)

    return output.T

def phase_correction(vector):
    """Phase correction to reduce distortions due to phase inconsistencies
    Args:
    vector: Beamforming vector with shape (..., bins, sensors).
    Returns: Phase corrected beamforming vectors. Lengths remain.
    """

    w = vector.copy()
    F, D = w.shape
    for f in range(1, F):
        w[f, :] *= np.exp(-1j * np.angle(
            np.sum(w[f, :] * w[f - 1, :].conj(), axis=-1, keepdims=True)))
    return w
def apply_beamforming_vector(vector, mix):
    return np.einsum('...a,...at->...t', vector.conj(), mix)

def gev_beamforming(audio_mix_spec_real,phase, irm_n_median, irm_t_median, gamma):
    Y_hat = gev_wrapper_on_masks(audio_mix_spec_real[0].transpose(1, 0, 2), irm_n_median, irm_t_median, True, 1e-1)
    Y_hat_time = istft_reconstruction_from_complex(Y_hat, phase)
    return Y_hat_time

def find_best_gamma(audio_mix_spec_real,phase, irm_n_median, irm_t_median, target_signal, noise_signal):
    best_gamma = None
    best_sdr = -np.inf

    for gamma in np.arange(0.01, 0.1, 0.01):
        Y_hat_time = gev_beamforming(audio_mix_spec_real, phase, irm_n_median, irm_t_median, gamma)
        sdr, sir, sar = getSeparationMetrics(Y_hat_time, noise_signal[:65535], target_signal[:65535], noise_signal[:65535])
        print(f"Gamma: {gamma}, SDR: {sdr}, SIR: {sir}, SAR: {sar}")

        if sdr > best_sdr:
            best_sdr = sdr
            best_gamma = gamma

    best_output = gev_beamforming(audio_mix_spec_real, phase,irm_n_median, irm_t_median, best_gamma)

    return best_gamma, best_sdr, best_output
def plot_psd_matrix(psd_matrix, title):
    """
    Plots the PSD matrix.
    Args:
        psd_matrix: PSD matrix with shape (bins, sensors, sensors)
        title: Title of the plot
    """
    avg_psd = np.abs(psd_matrix).mean(axis=-1)  # Average over sensors to reduce dimensions
    plt.imshow(avg_psd, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Sensors')
    plt.ylabel('Frequency Bins')
    plt.show()
def debug_psd(psd_matrix, title):
    print(f"{title} shape: {psd_matrix.shape}")
    plot_psd_matrix(psd_matrix, title)
def plot_psd_matrix_channels(psd_matrix, sensor_1, sensor_2, title):
    psd_specific = np.abs(psd_matrix[:, sensor_1, sensor_2])
    plt.imshow(psd_specific.T, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(f"{title} (Sensor {sensor_1} vs Sensor {sensor_2})")
    plt.xlabel('Frequency Bins')
    plt.ylabel('Time Frames')
    plt.show()

def plot_irm(irm, title):
    plt.figure(figsize=(10, 6))
    plt.imshow(irm, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.show()

def plot_spectrogram(spectrogram, title):
    plt.figure(figsize=(10, 6))
    plt.imshow(20 * np.log10(np.abs(spectrogram) + 1e-10), aspect='auto', origin='lower', cmap='magma')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.show()