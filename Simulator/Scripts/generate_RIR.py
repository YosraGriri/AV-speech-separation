import os
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np

def generate_rir(room):
    """
    Generates Room Impulse Response (RIR) using the image source model for a given room.

    Parameters:
    - room: Room object, representing the acoustic properties of the simulated room.

    Returns:
    - rir: numpy array, Simulated Room Impulse Response.
    """
    room.image_source_model()
    return room.compute_rir()


def plot_waveform(rir, fs):
    """
    Plots the waveform of the simulated Room Impulse Response (RIR).

    Parameters:
    - rir: numpy array, Simulated Room Impulse Response.
    - fs: int, Sampling frequency.
    """

    plt.figure()
    plt.plot(np.arange(len(rir)) / fs, rir)
    plt.title('Room Impulse Response Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


def save_processed_audio(room,
                         audio_filename,
                         output_folder):
    """
    Saves the audio with simulated room reverberation.

    Parameters:
    - room: Room object, representing the acoustic properties of the simulated room.
    - audio_filename: str, Path to the original audio file.
    - output_folder: str, Path to the folder where the processed audio will be saved.

    Returns:
    None
    """
    filename = f'{os.path.splitext(os.path.basename(audio_filename))[0]}_RIR.wav'
    print("The filename following:", filename)
    os.makedirs(output_folder, exist_ok=True)
    audio_reverb = room.mic_array.to_wav(
        os.path.join(output_folder, filename),
        norm=True,
        bitdepth=np.int16)  # Pass the RIR information



def read_random_wav_files(dataset_path):
    # Get a list of subfolders
    subfolders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]

    if len(subfolders) < 2:
        raise ValueError("There should be at least two subfolders in the dataset.")

    # Randomly select two different subfolders
    selected_subfolders = random.sample(subfolders, 2)

    # Get a list of wav files in each selected subfolder
    wav_files_subfolder1 = [f.path for f in os.scandir(selected_subfolders[0]) if f.is_file() and f.name.endswith('.wav')]
    wav_files_subfolder2 = [f.path for f in os.scandir(selected_subfolders[1]) if f.is_file() and f.name.endswith('.wav')]

    if not wav_files_subfolder1 or not wav_files_subfolder2:
        raise ValueError("No WAV files found in the selected subfolders.")

    # Randomly select one wav file from each subfolder
    selected_wav_file1 = random.choice(wav_files_subfolder1)
    selected_wav_file2 = random.choice(wav_files_subfolder2)

    # Read the wav files using librosa
    signal1, fs1 = librosa.load(selected_wav_file1, sr=None)
    signal2, fs2 = librosa.load(selected_wav_file2, sr=None)

    return signal1, fs1, signal2, fs2



