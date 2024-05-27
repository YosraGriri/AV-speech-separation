from utils.beamformer import *
from utils.signal_processing import *
from scipy.io import wavfile
import numpy as np
import os

def load_wav_files(base_path, file_prefix, mic_count, file_suffix):
    return [wavfile.read(os.path.join(base_path, f"{file_prefix}_mic{i}_{file_suffix}.wav"))[1] for i in range(mic_count)]

def load_npy_files(base_path, file_prefix, mic_count, file_suffix):
    return [np.load(os.path.join(base_path, f"{file_prefix}_mic{i}_{file_suffix}.npy")) for i in range(mic_count)]

base_path = ("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
             "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085")

# Loading audio files
target_files = load_wav_files(base_path, "00380", 4, "voice0")
noise_files = load_wav_files(base_path, "00085", 4, "voice1")
mix_files = [wavfile.read(os.path.join(base_path, f"00380_mic{i}_voice0_and_00085_mic{i}_voice1_mixed.wav"))[1] for i in range(4)]

# Loading mask files
noise_masks = load_npy_files(base_path, "00085", 4, "voice1_mask")
target_masks = load_npy_files(base_path, "00380", 4, "voice0_mask")

def generate_spectrogram_complex(audio, stft_frame, stft_hop, n_fft):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=n_fft, win_length=stft_frame, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel, real, imag

def istft_reconstruction_from_complex(real, imag, hop_length=160, win_length=400, length=65535):
    spec = real + 1j * imag
    wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length, length=length)
    return np.clip(wav, -1., 1.)

# Normalize and stack multi-channel audio data
multi_channel_mix = np.stack(mix_files, axis=0) / 32768
multi_channel_clean = np.stack(target_files, axis=0) / 32768
multi_channel_noise = np.stack(noise_files, axis=0) / 32768

# Generate spectrograms
audio_mix_spec, audio_mix_spec_real, audio_mix_spec_phase = generate_spectrogram_complex(multi_channel_mix, 400, 160, 512)
stft_clean, stft_clean_real, stft_clean_phase = generate_spectrogram_complex(multi_channel_clean, 400, 160, 512)
stft_noise, stft_noise_real, stft_noise_phase = generate_spectrogram_complex(multi_channel_noise, 400, 160, 512)

X_mix = np.stack(mix_files)

# Stack and process masks
noise_mask = np.stack([noise_mask[0] for noise_mask in noise_masks], axis=-1)
target_mask = np.stack([target_mask[0] for target_mask in target_masks], axis=-1)
print('Shape of target_mask:', target_mask.shape)

noise_mask_real = noise_mask[0, 0, :, :, :]
noise_mask_imag = noise_mask[0, 1, :, :, :]
target_mask_real = target_mask[0, 0, :, :, :]
target_mask_imag = target_mask[0, 1, :, :, :]

# Compute the magnitude of the masks
noise_mask_magnitude = np.sqrt(noise_mask_real ** 2 + noise_mask_imag ** 2)
target_mask_magnitude = np.sqrt(target_mask_real ** 2 + target_mask_imag ** 2)
print('Shape of noise_mask_magnitude:', noise_mask_magnitude.shape)

N_mask = np.median(noise_mask_magnitude, axis=-1)
X_mask = np.median(target_mask_magnitude, axis=-1)

N_mask = np.pad(N_mask, ((0, 1), (0, 0)), mode='constant', constant_values=1e-16)
X_mask = np.pad(X_mask, ((0, 1), (0, 0)), mode='constant', constant_values=1e-16)

Y = stft(X_mix, time_dim=1).transpose((1, 0, 2))
Y = Y[:256, :, :]
Y_hat = gev_wrapper_on_masks(Y, N_mask, X_mask)
Y_noise = gev_wrapper_on_masks(Y, X_mask, N_mask)
X_hat = istft(Y_hat[:, :], 512)
noise_hat = istft(Y_noise[:, :], 512)
X_GT = target_files[2]
noise_GT = noise_files[2]

def getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt):
    reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0)), axis=0)
    estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, False)
    return np.mean(sdr), np.mean(sir), np.mean(sar)

sdr, sir, sar = getSeparationMetrics(X_hat[:65280], noise_hat[:65280], X_GT[:65280], noise_GT[:65280])
print("SDR:", sdr)
print("SIR:", sir)
print("SAR:", sar)

#best_gamma, best_sdr, best_output = find_best_gamma(audio_mix_spec_real, stft_clean_phase[0, 0], irm_n_median, irm_t_median, X_GT, noise_GT)
#print(f'Best Gamma: {best_gamma}, Best SDR: {best_sdr}')
# Save the best output as 'youhou.wav'
#best_output_scaled = np.int16(best_output * 32767)
#wavfile.write('youhou.wav', sr, best_output_scaled)
#print('Best output saved as youhou.wav')
