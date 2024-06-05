# -*- coding: utf-8 -*-


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
target_path = "../data/VoxCeleb2/raw_audio_test/result/id08392_CZLQUQTssAE_00074VSid05202_A_BuBRwHT5w_00017/00017.wav"
noise_path = "../data/VoxCeleb2/raw_audio_test/result/id08392_CZLQUQTssAE_00074VSid05202_A_BuBRwHT5w_00017/noise.wav"
noise_speech = "../data/VoxCeleb2/raw_audio_test/result/id08392_CZLQUQTssAE_00074VSid05202_A_BuBRwHT5w_00017/00074.wav"

def mix_transform_and_stft_audio_files(file1, file2, matrix_A=None, duration_seconds=2.55, sample_rate=16000):
    duration = int(duration_seconds * sample_rate)
    
    # Read audio files
    sr_1, signal1 = wavfile.read(file1)
    sr_2, signal2 = wavfile.read(file2)
    
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
    
    # Calculate STFT for mixed signals, target signal, and noise signal
    Y = stft(mixed_signals, time_dim=1).transpose((1, 0, 2))
    stft_clean = stft(signal1[:duration])
    stft_noise = stft(signal2[:duration])
    
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


mixed_signals, target, noise, Y, stft_clean, stft_noise = mix_transform_and_stft_audio_files(target_path, noise_path)
irm_speech, irm_noise = get_irms(stft_clean, stft_noise)

Y_hat = gev_wrapper_on_masks(Y, irm_noise,irm_speech)
Y_noise_hat = gev_wrapper_on_masks(Y, irm_speech, irm_noise)
X_hat=istft(Y_hat[:,:],512)
X_noise=istft(Y_noise_hat[:,:],512)

sdr, sir , sar = getSeparationMetrics(X_hat[:duration], X_noise[:duration], target[:duration], noise[:duration])
print("SDR:", sdr)
print("SIR:", sir)
print("SAR:", sar)
pesq_score1 = pesq(16000, target[:duration], X_hat[:duration], 'wb')
print("PESQ Score:", pesq_score1)
