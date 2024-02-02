import matplotlib.pyplot as plt
import numpy as np
import os


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

