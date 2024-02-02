# Import necessary libraries
from simulate_room import create_room, add_microphone_array, add_sound_source, add_circular_array
from generate_RIR import generate_rir, plot_waveform, save_processed_audio
from plot import plot_room
import math
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
import numpy as np
import os

# Variables for creating the room

# Room dimensions in meters: Length, Width, Height
room_dimensions = [3.5, 3.5, 2.5]

# Absorption coefficient of the room
absorption_coefficient = 0.7

# Receiver (microphone) radius
receiver_radius = 0.5

# Number of rays for ray tracing
n_rays = 10000

# Energy threshold for ray tracing
energy_thres = 1e-5

# Desired reverberation time and dimensions of the room
rt60 = 0.3  # seconds

# Invert Sabine's formula to obtain parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60, room_dimensions)
materials = pra.Material(e_absorption)
print(e_absorption, max_order)

# Source directory for audio files
source_directory = "../Datasets/Grid Corpus/audio_25k/"

# Output folder for saving simulated RIRs
output_folder = "../Datasets/Grid Corpus/simulated_RIR"

# Iterate over subfolders in the source directory
for subfolder in os.listdir(source_directory):
    subfolder_path = os.path.join(source_directory, subfolder)

    # Check if the item in the directory is a subfolder
    if os.path.isdir(subfolder_path):
        print(f"Processing subfolder: {subfolder}")

        # Iterate over audio files in the subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".wav"):
                wav_path = os.path.join(subfolder_path, filename)
                print(f"Processing audio file: {wav_path}")

                # Read audio file
                fs, signal = wavfile.read(wav_path)

                # Source variables
                person_height = 1.7  # Height of a person (170 cm)
                center_x, center_y = room_dimensions[0] / 2, room_dimensions[1] / 2
                glasses_width = 0.139

                # Directory information
                current_directory = os.getcwd()
                project_directory = os.path.abspath(os.path.join(current_directory, "../"))

                # Speaker information
                speaker_position = [center_x + math.sqrt(2) / 2,
                                    center_y - math.sqrt(2) / 2,
                                    person_height]


                # Create the room using specified parameters
                shoebox = create_room(fs,
                                      shoebox=True,
                                      room_dimensions=room_dimensions,
                                      absorption_coefficient=absorption_coefficient,
                                      max_order=max_order,
                                      materials=materials,
                                      ray_tracing=True,
                                      air_absorption=True)

                # Set ray tracing parameters
                shoebox.set_ray_tracing(receiver_radius=receiver_radius,
                                        n_rays=n_rays,
                                        energy_thres=energy_thres)

                # Microphone array positions
                #mic_array = np.array([
                   #[center_x + glasses_width / 2, center_y + glasses_width / 2, person_height],
                   # [center_x + glasses_width / 2, center_y - glasses_width / 2, person_height],
                  #  [center_x - glasses_width / 2, center_y + glasses_width / 2, person_height],
                 #   [center_x - glasses_width / 2, center_y - glasses_width / 2, person_height]
                #])
                #print(mic_array.shape)

                # Add microphone array to the room
                #add_microphone_array(shoebox, mic_array)
                # Example usage of the function
                add_circular_array(shoebox,
                                   mic_center=np.array([0, 0, 1.7]),
                                   mic_radius=0.037,
                                   mic_n=4)

                # Add sound source to the room
                add_sound_source(shoebox, speaker_position, signal)

                # Perform image source model
                shoebox.image_source_model()

                # Plot the room layout
                plot_room(shoebox,
                          source_coordinates=speaker_position,
                          mic_coordinates=mic_array,
                          xlim=room_dimensions[0],
                          ylim=room_dimensions[1],
                          zlim=room_dimensions[2],
                          save_path=project_directory)

                # Plot room impulse response
                shoebox.plot_rir()

                # Compute room impulse response
                shoebox.compute_rir()

                # Simulate room acoustics
                shoebox.simulate()

                # Save processed audio with room impulse response in subfolder
                subfolder_output = os.path.join(output_folder, subfolder)
                save_processed_audio(shoebox, wav_path, subfolder_output)
