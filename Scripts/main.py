from simulate_room import create_room, add_microphone_array, add_sound_source
from generate_RIR import generate_rir, plot_waveform, save_processed_audio
from plot import plot_room
import math
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
import numpy as np
import os

### Variables for creating the room

room_dimensions = [3.5, 3.5, 2.5]  # Length, Width, Height in meters
absorption_coefficient = 0.7
receiver_radius = 0.5
n_rays = 10000
energy_thres = 1e-5
# The desired reverberation time and dimensions of the room
rt60 = 0.3  # seconds
# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60, room_dimensions)
materials = pra.Material(e_absorption)
print(e_absorption, max_order)
max_order = 10
print('-----------------------------------------------------------------------------')

# specify signal source
# Iteration over all files
source_directory = "C:/Users/yosra/Documents/AV-speech-separation/Datasets/Grid Corpus/audio_25k/s1/"
for filename in os.listdir(source_directory):
    if filename.endswith(".wav"):
        wav_path = os.path.join(source_directory, filename)
        print('-----------------------------------------------------------------------------')

        print(wav_path)
        print('-----------------------------------------------------------------------------')

        fs, signal = wavfile.read(wav_path)

        print(fs)
        print(signal.shape)
        ### Source variables
        # Height of a person (170 cm)
        person_height = 1.7
        center_x, center_y = room_dimensions[0] / 2, room_dimensions[1] / 2
        glasses_width = 0.139
        current_directory = os.getcwd()
        project_directory = os.path.abspath(os.path.join(current_directory, "../"))

        # Speaker information
        speaker_position = [center_x + math.sqrt(2) / 2,
                            center_y - math.sqrt(2) / 2,
                            person_height]
        print(f"Speaker Position: X = {speaker_position[0]}, "
              f"Y = {speaker_position[1]},"
              f" Z = {speaker_position[2]}")

        # Distance from the center to each microphone (assume glasses width of 20 cm)
        mic_array = np.array([

            [center_x + glasses_width / 2,
             center_y + glasses_width / 2,
             person_height],

            [center_x + glasses_width / 2,
             center_y - glasses_width / 2,
             person_height],

            [center_x - glasses_width / 2,
             center_y + glasses_width / 2,
             person_height],

            [center_x - glasses_width / 2,
             center_y - glasses_width / 2,
             person_height]
        ])

        shoebox = create_room(fs,
                              shoebox=True,
                              room_dimensions=room_dimensions,
                              room_corners=None,
                              absorption_coefficient=absorption_coefficient,
                              max_order=max_order,
                              materials=materials,
                              ray_tracing=True,
                              air_absorption=True)

        # Set the ray tracing parameters
        shoebox.set_ray_tracing(receiver_radius=receiver_radius,
                                n_rays=n_rays,
                                energy_thres=energy_thres)

        # Add microphone array
        add_microphone_array(shoebox,
                             mic_array
                             )

        # Add source

        add_sound_source(shoebox, speaker_position, signal)
        shoebox.image_source_model()

        plot_room(shoebox,
                  source_coordinates=speaker_position,
                  mic_coordinates=mic_array,
                  xlim=room_dimensions[0],
                  ylim=room_dimensions[1],
                  zlim=room_dimensions[2],
                  save_path=project_directory)
        shoebox.plot_rir()
        # Compute RIR
        shoebox.compute_rir()
        shoebox.simulate()


        # Save processed audio with RIR
        output_folder = "../Datasets/Grid Corpus/simulated_RIR"
        save_processed_audio(shoebox, wav_path, output_folder)
