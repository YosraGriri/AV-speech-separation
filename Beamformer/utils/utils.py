#Evaluation stuff
from pesq import pesq
from pystoi import stoi
import mir_eval
import numpy as np
import os
from scipy.io import wavfile
import glob
import librosa
import re

def load_wav_files(base_path):
    """
    Load WAV files according to specific criteria:
    - Files containing "voice0" and not "mixed" or "separated" as targets.
    - Files containing "voice1" and not "mixed" or "separated" as noise.
    - Files containing "mixed" as mixtures.
    
    :param base_path: The base directory to search for WAV files.
    :return: Three lists of loaded WAV files' data for targets, noise, and mixtures.
    """
    target_files = []
    noise_files = []
    mixture_files = []

    # Search for all wav files in the base path
    wav_files = glob.glob(os.path.join(base_path, "*.wav"))
    
    for file in wav_files:
        filename = os.path.basename(file)
        
        if "mixed" in filename:
            try:
                sr, data = wavfile.read(file)
                mixture_files.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        elif "voice0" in filename and "mixed" not in filename and "separated" not in filename:
            try:
                sr, data = wavfile.read(file)
                target_files.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        elif "voice1" in filename and "mixed" not in filename and "separated" not in filename:
            try:
                sr, data = wavfile.read(file)
                noise_files.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return target_files, noise_files, mixture_files

def load_numpy_files(base_path):
    """
    Load numpy files according to specific criteria:
    - Files containing "mask" and "voice0" as mask_targets.
    - Files containing "mask" and "voice1" as mask_noise.
    - All other numpy files as spec_mix.
    
    :param base_path: The base directory to search for numpy files.
    :return: Three lists of loaded numpy arrays for mask_targets, mask_noise, and spec_mix.
    """
    mask_targets = []
    mask_noise = []
    spec_mix_files = []

    # Search for all numpy files in the base path
    npy_files = glob.glob(os.path.join(base_path, "*.npy"))
    
    for file in npy_files:
        filename = os.path.basename(file)
        
        if "mask" in filename and "voice0" in filename:
            try:
                data = np.load(file)
                mask_targets.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        elif "mask" in filename and "voice1" in filename:
            try:
                data = np.load(file)
                mask_noise.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        else:
            try:
                data = np.load(file)
                spec_mix_files.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return mask_targets, mask_noise, spec_mix_files


def process_and_stack_masks(noise_masks, target_masks):
    """
    Processes and stacks noise and target masks, computes their magnitudes, and applies median and padding.

    :param noise_masks: List of noise masks numpy arrays.
    :param target_masks: List of target masks numpy arrays.
    :return: Processed target and noise mask magnitudes (X_mask and N_mask).
    """
    # Stack masks
    noise_mask = np.stack([noise_mask[0] for noise_mask in noise_masks], axis=-1)
    target_mask = np.stack([target_mask[0] for target_mask in target_masks], axis=-1)
    print('Shape of target_mask:', target_mask.shape)
    # Separate real and imaginary parts
    if len(noise_mask.shape) == 5:
        noise_mask = noise_mask[0]
        target_mask = target_mask[0]
    noise_mask_real = noise_mask[0, :, :, :]
    noise_mask_imag = noise_mask[1, :, :, :]
    target_mask_real = target_mask[0, :, :, :]
    target_mask_imag = target_mask[1, :, :, :]


    # Compute the magnitude of the masks
    noise_mask_magnitude = np.sqrt(noise_mask_real ** 2 + noise_mask_imag ** 2)
    target_mask_magnitude = np.sqrt(target_mask_real ** 2 + target_mask_imag ** 2)
    #print('Shape of noise_mask_magnitude:', noise_mask_magnitude.shape)

    # Compute median along the last axis
    N_mask = np.median(noise_mask_magnitude, axis=-1)
    X_mask = np.median(target_mask_magnitude, axis=-1)
    print('Shape of resulting target mask after median:', X_mask.shape)
    # Pad the masks
    N_mask = np.pad(N_mask, ((0, 1), (0, 0)), mode='constant', constant_values=1e-16)
    X_mask = np.pad(X_mask, ((0, 1), (0, 0)), mode='constant', constant_values=1e-16)

    return X_mask, N_mask

def mix_and_stft_audio_files(mixtures, targets, noises, 
                                       duration_seconds=2.55, sample_rate=16000):
    duration = int(duration_seconds * sample_rate)
    stft_frame = 400
    stft_hop = 160
    n_fft = 512
    
    # Normalize and stack multi-channel audio data
    multi_channel_mix = np.stack(mixtures, axis=0)[:,:duration] / 32768
    multi_channel_clean = np.stack(targets, axis=0)[:,:duration] / 32768
    multi_channel_noise = np.stack(noises, axis=0)[:,:duration] / 32768
    

    Y = np.array([librosa.core.stft(multi_channel_mix[ch], 
                                    hop_length=stft_hop, 
                                    n_fft=n_fft, win_length=stft_frame, 
                                    center=True) 
                  for ch in range(multi_channel_mix.shape[0])])
    
    Y = Y.transpose((1, 0, 2))  # Transpose to (frequency_bins, channels, time_frames)
   
    stft_clean = np.array([librosa.core.stft(multi_channel_clean[ch], 
                                             hop_length=stft_hop, 
                                             n_fft=n_fft, 
                                             win_length=stft_frame, 
                                             center=True
                                             ) 
                           for ch in range(multi_channel_clean.shape[0])])
    stft_noise = np.array([librosa.core.stft(multi_channel_noise[ch],
                                             hop_length=stft_hop, 
                                             n_fft=n_fft, 
                                             win_length=stft_frame, 
                                             center=True) 
                           for ch in range(multi_channel_noise.shape[0])])

    
    return multi_channel_mix, multi_channel_clean, multi_channel_noise, Y, stft_clean, stft_noise

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

def getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt):
    reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0)), axis=0)
    estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, False)
    return np.mean(sdr), np.mean(sir), np.mean(sar)

def extract_ids(filename):
    # Extract the last 5 digits as noise_id
    noise_id = filename[-5:]
    
    # Regular expression to find the target_id which is the 5 digits before 'VS'
    match = re.search(r'(\d{5})VS', filename)
    
    if match:
        target_id = match.group(1)
    else:
        target_id = None

    return target_id, noise_id
