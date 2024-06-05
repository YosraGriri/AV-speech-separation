import matplotlib.pyplot as plt
import numpy as np
import wave
import librosa
from scipy.io import wavfile

#Evaluation stuff
from pesq import pesq
from pystoi import stoi
import mir_eval

import sys
sys.path.append('../utils')
from utils.signal_processing import*
from utils.beamform_it import *

duration = int(2.55 * 16000)
stft_frame = 400
stft_hop = 160
n_fft = 512


target_path = "../data/VoxCeleb2/raw_audio_test/result/id08392_CZLQUQTssAE_00074VSid05202_A_BuBRwHT5w_00017/00017.wav"
noise_path = "../data/VoxCeleb2/raw_audio_test/result/id08392_CZLQUQTssAE_00074VSid05202_A_BuBRwHT5w_00017/noise.wav"
noise_speech_path = "../data/VoxCeleb2/raw_audio_test/result/id08392_CZLQUQTssAE_00074VSid05202_A_BuBRwHT5w_00017/00074.wav"

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

def getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt):
    reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0)), axis=0)
    estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, False)
    return np.mean(sdr), np.mean(sir), np.mean(sar)


def get_irms(stft_clean, stft_noise):
    mag_clean = np.abs(stft_clean) ** 2
    mag_noise = np.abs(stft_noise) ** 2
    irm_speech = mag_clean / (mag_clean + mag_noise+1e-16)
    print(irm_speech.shape)
    irm_noise = mag_noise / (mag_clean + mag_noise+1e-16)
    return irm_speech[:, :], irm_noise[:, :]

def inverse_stft(spec, hop_length, win_length, length):
    """
    Performs inverse STFT using librosa and clips the output to the range [-1, 1].
    :param spec: STFT spectrogram
    :param hop_length: Hop length for inverse STFT
    :param win_length: Window length for inverse STFT
    :param length: Length of the output signal
    :return: Time-domain signal
    """
    wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length, length=length)
    return np.clip(wav, -1., 1.)

mixed_signals, target, noise, Y, stft_clean, stft_noise = mix_transform_and_stft_audio_files(target_path, noise_path)
irm_speech, irm_noise = get_irms(stft_clean, stft_noise)

mixed_signals_speakers, target, noise_speech, Y_speakers, stft_clean_speakers, stft_noise_speakers = mix_transform_and_stft_audio_files(target_path, noise_speech_path)
irm_speech_s1, irm_noise_s2 = get_irms(stft_clean_speakers, stft_noise_speakers)

Y_hat = gev_wrapper_on_masks(Y, irm_noise,irm_speech)
Y_noise_hat = gev_wrapper_on_masks(Y, irm_speech, irm_noise)
X_hat = librosa.istft(Y_hat, hop_length=160, win_length=400)
X_noise = librosa.istft(Y_noise_hat, hop_length=160, win_length=400)

Y_hat_speakers = gev_wrapper_on_masks(Y_speakers, irm_noise_s2,irm_speech_s1)
Y_noise_hat_speakers = gev_wrapper_on_masks(Y_speakers, irm_speech_s1, irm_noise_s2)
X_hat_speakers = librosa.istft(Y_hat_speakers, hop_length=160, win_length=400)
X_noise_speakers = librosa.istft(Y_noise_hat_speakers, hop_length=160, win_length=400)

sdr, sir , sar = getSeparationMetrics(X_hat[:duration], X_noise[:duration], target[:duration], noise[:duration])
print("SDR:", sdr)
print("SIR:", sir)
print("SAR:", sar)
pesq_score1 = pesq(16000, target[:duration], X_hat[:duration], 'wb')
print("PESQ Score:", pesq_score1)

print('--------------------------------')
print('--------------------------------')
print('--------------------------------')

sdr, sir , sar = getSeparationMetrics(X_hat_speakers[:duration], X_noise_speakers[:duration], target[:duration], noise_speech[:duration])
print("SDR:", sdr)
print("SIR:", sir)
print("SAR:", sar)
pesq_score1 = pesq(16000, target[:duration], X_hat_speakers[:duration], 'wb')
print("PESQ Score:", pesq_score1)