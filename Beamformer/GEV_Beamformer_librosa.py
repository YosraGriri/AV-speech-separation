import matplotlib.pyplot as plt
import numpy as np
import wave
import librosa
from scipy.io import wavfile
import soundfile as sf
#Evaluation stuff
from pesq import pesq
from pystoi import stoi
import mir_eval

import sys
sys.path.append('../utils')
from utils.signal_processing import*
from utils.beamform_it import *
from utils.utils import*

duration = int(2.55 * 16000)
stft_frame = 400
stft_hop = 160
n_fft = 512


target_path ="../data/test/results/original/complex/id04030_7mXUMuo5_NE_00002VSid06913_Y60BPD7Ao1U_00089/00002.wav"
noise_path = "../data/test/results/original/complex/noise.wav"
noise_speech_path ="../data/test/results/original/complex/id04030_7mXUMuo5_NE_00002VSid06913_Y60BPD7Ao1U_00089/00089.wav"

target_mask = np.load("../data/test/results/original/complex/id04030_7mXUMuo5_NE_00002VSid06913_Y60BPD7Ao1U_00089/00002_mask.npy")
noise_mask = np.load("../data/test/results/original/complex/id04030_7mXUMuo5_NE_00002VSid06913_Y60BPD7Ao1U_00089/00089_mask.npy")
noise_mask_real = noise_mask[0, 0, :, :]
noise_mask_imag = noise_mask[0, 1, :, :]
target_mask_real = target_mask[0, 0, :, :]
target_mask_imag = target_mask[0, 1, :, :]
noise_mask_magnitude = np.sqrt(noise_mask_real ** 2 + noise_mask_imag ** 2)
target_mask_magnitude = np.sqrt(target_mask_real ** 2 + target_mask_imag ** 2)
X_mask = target_mask_magnitude
N_mask = noise_mask_magnitude
N_mask = np.pad(N_mask, ((0, 1), (0, 0)), mode='constant', constant_values=1e-16)
X_mask = np.pad(X_mask, ((0, 1), (0, 0)), mode='constant', constant_values=1e-16)


def mix_transform_and_stft_audio_files(file1, file2, matrix_A=None, duration_seconds=2.55, sample_rate=16000):
    duration = int(duration_seconds * sample_rate)
    
    # Read audio files
    sr_1, signal1 = wavfile.read(file1)
    sr_2, signal2 = wavfile.read(file2)
    
    signal1 = signal1/32768
    signal2 = signal2/32768
    
    # Validate that sample rates are the same
    assert sr_1 == sr_2, "Sample rates do not match."
    
    # Define the default transformation matrix if none is provided
    if matrix_A is None:
        matrix_A = np.array([[1, 1], [1, -1]])
    
    # Validate the shape of the transformation matrix
    assert matrix_A.shape == (2, 2), "Transformation matrix must be 2x2."
    
    # Stack signals and apply the transformation
    signals = np.stack([signal1[:duration], signal2[:duration]])
    mixed_signals = matrix_A @ signals
    
    Y = np.array([librosa.core.stft(mixed_signals[ch], hop_length=stft_hop, n_fft=n_fft, win_length=stft_frame, center=True) 
                  for ch in range(mixed_signals.shape[0])])
    Y = Y.transpose((1, 0, 2))  # Transpose to (frequency_bins, channels, time_frames)
    stft_clean = librosa.core.stft(signal1[:duration], hop_length=stft_hop, n_fft=n_fft, win_length=stft_frame, center=True)
    stft_noise = librosa.core.stft(signal2[:duration], hop_length=stft_hop, n_fft=n_fft, win_length=stft_frame, center=True)

    
    return mixed_signals, signal1[:duration], signal2[:duration], Y, stft_clean, stft_noise



mixed_signals, target, noise, Y, stft_clean, stft_noise = mix_transform_and_stft_audio_files(target_path, noise_path)
irm_speech, irm_noise = get_irms(stft_clean, stft_noise)
sf.write("../data/test/results/original/mixed_signals_noisy.wav", mixed_signals.T[:duration], 16000)
mixed_signals_speakers, target, noise_speech, Y_speakers, stft_clean_speakers, stft_noise_speakers = mix_transform_and_stft_audio_files(target_path, noise_speech_path)
irm_speech_s1, irm_noise_s2 = get_irms(stft_clean_speakers, stft_noise_speakers)
sf.write("../data/test/results/original/mixed_signals_id04030_7mXUMuo5_NE_00002VSid06913_Y60BPD7Ao1U_00089.wav", 
         mixed_signals_speakers.T[:duration], 16000)
Y_hat = gev_wrapper_on_masks(Y, irm_noise,irm_speech)
Y_noise_hat = gev_wrapper_on_masks(Y, irm_speech, irm_noise)
X_hat = librosa.istft(Y_hat, hop_length=160, win_length=400)
X_noise = librosa.istft(Y_noise_hat, hop_length=160, win_length=400)

Y_hat_speakers = gev_wrapper_on_masks(Y_speakers, irm_noise_s2, irm_speech_s1)
Y_noise_hat_speakers = gev_wrapper_on_masks(Y_speakers, irm_speech_s1, irm_noise_s2)
X_hat_speakers = librosa.istft(Y_hat_speakers, hop_length=160, win_length=400)
X_noise_speakers = librosa.istft(Y_noise_hat_speakers, hop_length=160, win_length=400)


Y_hat_speakers_visvoi = gev_wrapper_on_masks(Y_speakers, N_mask, X_mask)
Y_noise_hat_speakers_visvoi = gev_wrapper_on_masks(Y_speakers, X_mask, N_mask)
X_hat_speakers_visvoi = librosa.istft(Y_hat_speakers_visvoi, hop_length=160, win_length=400)
X_noise_speakers_visvoi = librosa.istft(Y_noise_hat_speakers_visvoi, hop_length=160, win_length=400)

print('Results from speech + noise')
sdr, sir , sar = getSeparationMetrics(X_hat[:duration], X_noise[:duration], target[:duration], noise[:duration])
print("SDR:", sdr)
print("SIR:", sir)
print("SAR:", sar)
stoi_score1 = stoi(target[:duration], X_hat[:duration], 16000, extended=False)
stoi_score2 = stoi(noise[:duration], X_noise[:duration], 16000, extended=False)
stoi_score = (stoi_score1 + stoi_score2) / 2
print(f'STOI score: {stoi_score}')

pesq_score1 = pesq(16000, target[:duration], X_hat[:duration], 'wb')
#pesq_score2 = pesq(16000, noise[:duration], X_noise[:duration], 'wb')
pesq_score =pesq_score1
print(f'PESQ score is: {pesq_score}')
sf.write("../data/test/results/original/GEV_00002_denoise.wav", X_hat[:duration], 16000)

print('Results from speech + speech')
print('--------------------------------')
print('--------------------------------')

sdr, sir , sar = getSeparationMetrics(X_hat_speakers[:duration], X_noise_speakers[:duration], target[:duration], noise_speech[:duration])
print("SDR:", sdr)
print("SIR:", sir)
print("SAR:", sar)
stoi_score1 = stoi(target[:duration], X_hat_speakers[:duration], 16000, extended=False)
stoi_score2 = stoi(noise_speech[:duration], X_noise_speakers[:duration], 16000, extended=False)
stoi_score = (stoi_score1 + stoi_score2) / 2
print(f'STOI score: {stoi_score}')

pesq_score1 = pesq(16000, target[:duration], X_hat_speakers[:duration], 'wb')
pesq_score2 = pesq(16000, noise_speech[:duration], X_noise_speakers[:duration], 'wb')
pesq_score = (pesq_score1 + pesq_score2) / 2
print(f'PESQ score is: {pesq_score}')
sf.write("../data/test/results/original/GEV_00002_sep.wav", X_hat_speakers[:duration], 16000)
sf.write("../data/test/results/original/GEV_00089_sep.wav", X_noise_speakers[:duration], 16000)
print('Results from speech + speech Using VisualVoice')
print('--------------------------------')
print('--------------------------------')

sdr, sir , sar = getSeparationMetrics(X_hat_speakers_visvoi[:duration], X_noise_speakers_visvoi[:duration], target[:duration], noise_speech[:duration])
print("SDR:", sdr)
print("SIR:", sir)
print("SAR:", sar)
stoi_score1 = stoi(target[:duration], X_hat_speakers_visvoi[:duration], 16000, extended=False)
stoi_score2 = stoi(noise_speech[:duration], X_noise_speakers_visvoi[:duration], 16000, extended=False)
stoi_score = (stoi_score1 + stoi_score2) / 2
print(f'STOI score: {stoi_score}')

pesq_score1 = pesq(16000, target[:duration], X_hat_speakers_visvoi[:duration], 'wb')
pesq_score2 = pesq(16000, noise_speech[:duration], X_noise_speakers_visvoi[:duration], 'wb')
pesq_score = (pesq_score1 + pesq_score2) / 2
print(f'PESQ score is: {pesq_score}')
