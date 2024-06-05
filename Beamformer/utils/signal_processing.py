# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:31:02 2024

@author: yosra
"""
import numpy as np
import scipy
from numpy.fft import rfft, irfft
from scipy import signal
from scipy.io.wavfile import write as wav_write
from scipy.linalg import LinAlgError
import string
import threading

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    if axis is None:
        a = np.ravel(a)
        axis = 0

    l = a.shape[axis]
    if overlap >= length:
        raise ValueError("frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0:
        raise ValueError("overlap must be nonnegative and length must be positive")

    step = length - overlap
    if end == 'cut':
        n = (l - overlap) // step
    elif end == 'pad':
        n = (l + step - 1) // step
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (0, n * step + overlap - l)
        a = np.pad(a, pad_width, mode='constant', constant_values=endvalue)
    elif end == 'wrap':
        n = (l + step - 1) // step
        wrap_width = n * step + overlap - l
        a = np.concatenate([a, a.take(range(wrap_width), axis=axis)], axis=axis)

    shape = list(a.shape)
    shape[axis] = length
    shape.insert(axis, n)

    strides = list(a.strides)
    strides.insert(axis, step * a.strides[axis])

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



def stft(time_signal, time_dim=None, size=512, shift=256, window=signal.blackman, fading=True, window_length=None):
    if time_dim is None:
        time_dim = np.argmax(time_signal.shape)
    
    if fading:
        pad_width = [(0, 0)] * time_signal.ndim
        pad_width[time_dim] = (size - shift, size - shift)
        time_signal = np.pad(time_signal, pad_width, mode='constant')

    frames = int(np.ceil((time_signal.shape[time_dim] - size + shift) / shift))
    samples = frames * shift + size - shift
    pad_width = [(0, 0)] * time_signal.ndim
    pad_width[time_dim] = (0, samples - time_signal.shape[time_dim])
    time_signal = np.pad(time_signal, pad_width, mode='constant')

    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = np.pad(window, (0, size - window_length), mode='constant')

    time_signal_seg = segment_axis(time_signal, size, size - shift, axis=time_dim)
    return np.fft.rfft(time_signal_seg * window, axis=time_dim + 1)


def _biorthogonal_window_loopy(analysis_window, shift):
    fft_size = len(analysis_window)
    assert fft_size % shift == 0, "FFT size must be a multiple of shift!"
    number_of_shifts = fft_size // shift

    sum_of_squares = np.zeros(shift)
    for synthesis_index in range(shift):
        for sample_index in range(number_of_shifts + 1):
            analysis_index = synthesis_index + sample_index * shift
            if analysis_index < fft_size:
                sum_of_squares[synthesis_index] += analysis_window[analysis_index] ** 2

    sum_of_squares = np.tile(sum_of_squares, number_of_shifts)
    synthesis_window = analysis_window / (sum_of_squares * fft_size)
    return synthesis_window

def istft(stft_signal, size=512, shift=256, window=signal.blackman, fading=True, window_length=None):
    assert stft_signal.shape[1] == size // 2 + 1, "STFT signal shape mismatch"

    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = np.pad(window, (0, size - window_length), mode='constant')

    window = _biorthogonal_window_loopy(window, shift)
    window *= size

    time_signal = np.zeros(stft_signal.shape[0] * shift + size - shift)

    for j in range(stft_signal.shape[0]):
        i = j * shift
        time_signal[i:i + size] += window * np.real(irfft(stft_signal[j]))

    if fading:
        time_signal = time_signal[size - shift:-(size - shift)]

    return time_signal