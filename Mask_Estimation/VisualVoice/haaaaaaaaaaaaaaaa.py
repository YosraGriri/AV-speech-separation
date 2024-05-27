from utils.beamformer import *
from utils.signal_processing import *
from scipy.io import wavfile


# Loading audio files
_, target_0 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                            "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic0_voice0.wav")
_, target_1 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                            "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic1_voice0.wav")
_, target_2 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                            "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic2_voice0.wav")
_, target_3 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                            "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic3_voice0.wav")

_, noise_0 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                           "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic0_voice1.wav")
_, noise_1 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                           "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic1_voice1.wav")
_, noise_2 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                           "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic2_voice1.wav")
_, noise_3 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                           "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic3_voice1.wav")

_, mix_0 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                         "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/"
                         "00380_mic0_voice0_and_00085_mic0_voice1_mixed.wav")
_, mix_1 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                         "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/"
                         "00380_mic1_voice0_and_00085_mic1_voice1_mixed.wav")
_, mix_2 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                         "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/"
                         "00380_mic2_voice0_and_00085_mic2_voice1_mixed.wav")
_, mix_3 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                         "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/"
                         "00380_mic3_voice0_and_00085_mic3_voice1_mixed.wav")


### Loading data
noise_mask_0 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                       "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic0_voice1_mask.npy")
noise_mask_1 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                       "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic1_voice1_mask.npy")
noise_mask_2 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                       "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic2_voice1_mask.npy")
noise_mask_3 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                       "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic3_voice1_mask.npy")

target_mask_0 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                        "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic0_voice0_mask.npy")
target_mask_1 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                        "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic1_voice0_mask.npy")
target_mask_2 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                        "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic2_voice0_mask.npy")
target_mask_3 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                        "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic3_voice0_mask.npy")

def generate_spectrogram_complex_(audio, stft_frame, stft_hop, n_fft):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=n_fft, win_length=stft_frame, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel, real, imag

def istft_reconstruction_from_complex_(real, imag, hop_length=160, win_length=400, length=65535):
    spec = real + 1j*imag
    wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length, length=length)
    return np.clip(wav, -1., 1.)

multi_channel_mix = np.stack((mix_0, mix_1, mix_2, mix_3), axis=0)
multi_channel_mix = multi_channel_mix/32768 #normalization 16 bit (-1,1)
multi_channel_clean = np.stack((target_0, target_1, target_2, target_3), axis=0)
multi_channel_clean = multi_channel_clean/32768
multi_channel_noise = np.stack((noise_0, noise_1, noise_2, noise_3), axis=0)
multi_channel_noise = multi_channel_noise /32768

audio_mix_spec , audio_mix_spec_real, audio_mix_spec_phase = generate_spectrogram_complex_(multi_channel_mix, 400, 160, 512)
stft_clean, stft_clean_real, stft_clean_phase = generate_spectrogram_complex_(multi_channel_clean, 400, 160, 512)
stft_noise, stft_noise_real, stft_noise_phase = generate_spectrogram_complex_(multi_channel_noise, 400, 160, 512)

irm_t, irm_n = get_irms(stft_clean_real, stft_noise_real)
pred_spec_1_real = audio_mix_spec[0, 0, :] * irm_t[0] - audio_mix_spec[0, 1, :] * irm_t[0]
pred_spec_1_imag = audio_mix_spec[0, 1, :] * irm_t[0] + audio_mix_spec[0, 0, :] * irm_t[0]

### Now we can apply beamforming.
irm_n_median = np.median(irm_n, axis=1)
irm_t_median = np.median(irm_t, axis=1)
#print(f'Shape of irm_n_median is: {irm_n_median.shape}')
#print(f'Shape of irm_n_median is: {irm_n_median.shape}')

#Y_hat = gev_beamforming(audio_mix_spec_real, stft_clean_phase[0, 0], irm_n_median, irm_t_median, 1e-1)
#Y_noise_hat =gev_beamforming(audio_mix_spec_real, stft_noise_phase[0, 0], irm_n_median, irm_t_median, 1e-1)


#best_gamma, best_sdr, best_output = find_best_gamma(audio_mix_spec_real, stft_clean_phase[0, 0],irm_n_median, irm_t_median, target_0, noise_0)
#print(f'Best Gamma: {best_gamma}, Best SDR: {best_sdr}')

X_mix = np.stack([mix_0, mix_1, mix_2, mix_3])

noise_mask = np.stack((noise_mask_0[0], noise_mask_1[0], noise_mask_2[0], noise_mask_3[0]), axis=-1)
target_mask = np.stack((target_mask_0[0], target_mask_1[0], target_mask_2[0], target_mask_3[0]), axis=-1)
print('Shape of target_mask:', target_mask.shape)
noise_mask_real = noise_mask[0,0, :, :,:]
noise_mask_imag = noise_mask[0,1, :, :,:]
target_mask_real = target_mask[0,0, :, :,:]
target_mask_imag = target_mask[0,1, :, :,:]

# Compute the magnitude of the masks
noise_mask_magnitude = np.sqrt(noise_mask_real**2 + noise_mask_imag**2)
target_mask_magnitude = np.sqrt(target_mask_real**2 + target_mask_imag**2)
print('Shape of noise_mask_magnitude:', noise_mask_magnitude.shape)

N_mask = np.median(noise_mask_magnitude, axis=-1)
X_mask = np.median(target_mask_magnitude, axis=-1)

N_mask = np.pad(N_mask, ((0, 1), (0, 0)), mode='constant', constant_values=1e-16)
X_mask = np.pad(X_mask, ((0, 1), (0, 0)), mode='constant', constant_values=1e-16)

Y = stft(X_mix, time_dim=1).transpose((1, 0, 2))
Y = Y[:256, :, :]
Y_hat = gev_wrapper_on_masks(Y, N_mask, X_mask)
Y_noise = gev_wrapper_on_masks(Y, X_mask, N_mask)
X_hat =istft(Y_hat[:, :],512)
noise_hat = istft(Y_noise[:, :], 512)
X_GT = target_2
noise_GT = noise_2
sdr, sir , sar = getSeparationMetrics(X_hat[:65280], noise_hat[:65280], X_GT[:65280], noise_GT[:65280])
print("SDR:", sdr)
print("SIR:", sir)
print("SAR:", sar)
#pesq_score1 = pesq(fs, X_GT[:65280], X_hat[:65280], 'wb')
#print("PESQ Score:", pesq_score1)
#best_gamma, best_sdr, best_output = find_best_gamma(audio_mix_spec_real,
 #                                                   stft_clean_phase[0, 0],
  #                                                  irm_n_median,
   #                                                 irm_t_median,
    #                                                X_GT,
     #                                               noise_GT)
