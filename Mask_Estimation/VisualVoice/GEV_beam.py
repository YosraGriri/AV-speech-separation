from utils.beamformer import *
from utils.signal_processing import *
from scipy.io import wavfile
from scipy.signal import stft, istft

# Loading audio files
sr, target_0 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                            "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic0_voice0.wav")
sr, target_1 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                            "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic1_voice0.wav")
sr, target_2 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                            "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic2_voice0.wav")
sr, target_3 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                            "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic3_voice0.wav")

sr, noise_0 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                           "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic0_voice1.wav")
sr, noise_1 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                           "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic1_voice1.wav")
sr, noise_2 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                           "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic2_voice1.wav")
sr, noise_3 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                           "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic3_voice1.wav")

sr, mix_0 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                         "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/"
                         "00380_mic0_voice0_and_00085_mic0_voice1_mixed.wav")
sr, mix_1 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                         "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/"
                         "00380_mic1_voice0_and_00085_mic1_voice1_mixed.wav")
sr, mix_2 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                         "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/"
                         "00380_mic2_voice0_and_00085_mic2_voice1_mixed.wav")
sr, mix_3 = wavfile.read("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                         "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/"
                         "00380_mic3_voice0_and_00085_mic3_voice1_mixed.wav")

## loading masks

target_Mask0 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                       "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic0_voice0_mask.npy")
target_Mask1 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                       "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic1_voice0_mask.npy")
target_Mask2 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                       "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic2_voice0_mask.npy")
target_Mask3 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                       "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00380_mic3_voice0_mask.npy")

noise_Mask0 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                      "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic0_voice1_mask.npy")
noise_Mask1 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                      "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic1_voice1_mask.npy")
noise_Mask2 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                      "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic2_voice1_mask.npy")
noise_Mask3 = np.load("E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
                      "id04657_paq5LdKsJeM_00380VSid06209_UfVs86X6YEo_00085/00085_mic3_voice1_mask.npy")

multi_channel_mix = np.stack((mix_0, mix_1, mix_2, mix_3), axis=0)
multi_channel_clean = np.stack((target_0, target_1, target_2, target_3), axis=0)
multi_channel_noise = np.stack((noise_0, noise_1, noise_2, noise_3), axis=0)

Y = stft(multi_channel_mix, time_dim=1).transpose((1, 0, 2))
Y = Y[:256, :, :]
stft_clean = stft(multi_channel_clean, time_dim=1).transpose((1, 0, 2))
stft_noise = stft(multi_channel_noise, time_dim=1).transpose((1, 0, 2))

noise_mask = np.stack((noise_Mask0[0], noise_Mask1[0], noise_Mask2[0], noise_Mask3[0]), axis=-1)
target_mask = np.stack((target_Mask0[0], target_Mask1[0], target_Mask2[0], target_Mask3[0]), axis=-1)
print('Shape of target_mask:', target_mask.shape)
noise_mask_real = noise_mask[0, 0, :, :, :]
noise_mask_imag = noise_mask[0, 1, :, :, :]
target_mask_real = target_mask[0, 0, :, :, :]
target_mask_imag = target_mask[0, 1, :, :, :]

# Compute the magnitude of the masks
noise_mask_magnitude = np.sqrt(noise_mask_real ** 2 + noise_mask_imag ** 2)
target_mask_magnitude = np.sqrt(target_mask_real ** 2 + target_mask_imag ** 2)
print('Shape of target_mask_magnitude:', target_mask_magnitude.shape)

N_mask = np.median(noise_mask_magnitude, axis=-1)
X_mask = np.median(target_mask_magnitude, axis=-1)
N_mask = np.pad(N_mask, ((0, 1), (0, 0)), mode='constant', constant_values=1e-16)
X_mask = np.pad(X_mask, ((0, 1), (0, 0)), mode='constant', constant_values=1e-16)
print(f'Shape of X_mask is: {X_mask.shape}')
Y_hat = gev_wrapper_on_masks(Y, N_mask, X_mask)
Y_noise = gev_wrapper_on_masks(Y, X_mask, N_mask)

# Convert to time domain
Y_hat_time = istft(Y_hat, 512)
Y_noise_time = istft(Y_noise, 512)

# Plot time domain signals
#plt.figure()
#plt.plot(Y_hat_time)
#plt.title("Beamformed Target Signal")
#plt.show()

# Plot time domain signals
#plt.figure()
#plt.plot(target_0)
#plt.title("Target Signal channel 0")
#plt.show()

stft_clean = stft(multi_channel_clean, time_dim=1).transpose((1, 0, 2))
stft_noise = stft(multi_channel_noise, time_dim=1).transpose((1, 0, 2))
Y = stft(multi_channel_mix, time_dim=1).transpose((1, 0, 2))
irm_t, irm_n = get_irms(stft_clean, stft_noise)
print("Shape of irm_t:", irm_t.shape)  # Should print (333, 4, 257)
irm_t = np.median(irm_t, axis=1)
irm_n = np.median(irm_n, axis=1)
print("Shape of irm_t_median:", irm_t.shape)  # Should print (333, 257)

Y_hatttt = gev_wrapper_on_masks(Y, irm_n.T, irm_t.T)
Y_noiseee = gev_wrapper_on_masks(Y, irm_t.T, irm_n.T)
#plt.figure()
#plt.plot(Y_hatttt)
#plt.title("Beamformed GT IRM")
#plt.show()


plot_irm(irm_t, "Speech IRM Median")
plot_irm(irm_n, "Noise IRM Median")

