import argparse
import soundfile as sf
from simulate_room import *
from plot import *
from get_voice import get_voices
import json
from collections import defaultdict
from utils import *
import os
from pathlib import Path
import numpy as np

# Mean and STD of the signal peak
FG_VOL_MIN = 0.15
FG_VOL_MAX = 0.4
BG_VOL_MIN = 0.2
BG_VOL_MAX = 0.5

def generate_room_simulations(args):
    """
    Generate room simulations for all folders with voice 0.

    Parameters:
    - args: argparse.Namespace, Command-line arguments and parameters for sample generation.
    """
    folders = [f for f in os.listdir(args.source_dir) if os.path.isdir(os.path.join(args.source_dir, f))]

    for idx, folder in enumerate(folders[19:21]):
        args.folder = folder
        print(f"Generating room simulation for folder: {folder}")
        generate_sample_vox(args, folder, idx)  # Use folder index for unique identification




def generate_sample_vox(args, folder, idx: int) -> str:
    folder_path = os.path.join(args.source_dir, folder)

    voices_data, video_paths, voice_durations = get_voices(args, folder_path)
    print(f'Number of video paths: {len(video_paths)}')
    print(f'Video paths: {video_paths}')

    e_absorption, max_order = pra.inverse_sabine(args.rt60, args.room_dimensions)
    if args.max_order is not None:
        max_order = args.max_order
    materials = pra.Material(e_absorption)

    for voice_idx, (voice, duration) in enumerate(zip(voices_data, voice_durations)):
        print(f'Processing voice {voice_idx} with duration {duration}s')

        video_path = video_paths[voice_idx]
        # Extract the actual voice signal assuming it's the first element
        voice_signal = voice[0] if isinstance(voice, (list, tuple)) else voice

        # Ensure the voice is a numpy array and has the correct shape
        voice_signal = np.array(voice_signal)

        if len(voice_signal.shape) != 1:
            raise ValueError(f"Expected a 1D array for the voice signal, got shape {voice_signal.shape}")

        room = create_room(args.sr,
                           shoebox=True,
                           room_dimensions=args.room_dimensions,
                           absorption_coefficient=e_absorption,
                           max_order=max_order,
                           materials=materials,
                           ray_tracing=True,
                           air_absorption=True)

        mic_center = [x * 0.5 for x in args.room_dimensions[:2]]
        mic_center.append(args.avg_human_height)
        print(f'The mic center coordinates are: {mic_center}')
        total_samples = int(duration * args.sr)
        mic_dim = add_circular_array(room, mic_center, args.mic_radius, args.n_mics)
        max_radius = min(args.room_dimensions[0], args.room_dimensions[1]) / 2
        voice_idx = 1

        voice_theta = np.random.uniform(low=0, high=2 * np.pi) if args.random else np.radians(
            args.angle_between_sources * voice_idx)
        voice_radius = np.random.uniform(low=0, high=max_radius) if args.random else max_radius / 2

        voice_loc = [
            voice_radius * np.cos(voice_theta) + mic_center[0],
            voice_radius * np.sin(voice_theta) + mic_center[1],
            args.avg_human_height
        ]

        voice_loc = [
            max(0, min(voice_loc[0], args.room_dimensions[0])),
            max(0, min(voice_loc[1], args.room_dimensions[1])),
            args.avg_human_height
        ]

        print(f'The voice has this location: {voice_loc}')
        1/0
        room.add_source(voice_loc, signal=voice_signal)


        room.image_source_model()
        room.simulate()
        fg_signals = room.mic_array.signals[:, :total_samples]

        fg_target = np.random.uniform(FG_VOL_MIN, FG_VOL_MAX)
        fg_signals = fg_signals * fg_target / abs(fg_signals).max()

        # Extract the speaker ID and video ID from the folder path
        path_parts = video_path.split('\\')

        speaker_id = path_parts[-3]
        segment = path_parts[-2]
        wav_id = path_parts[-1]

        # Save the simulated microphone signals to the specified path
        for mic_idx in range(args.n_mics):
            filename = f"{wav_id}_mic{mic_idx}_voice{voice_idx}.wav"
            mic_simulated_dir = Path(args.output_path) / speaker_id / segment / filename
            print(f'Saving to {mic_simulated_dir}')
            mic_simulated_dir.parent.mkdir(parents=True, exist_ok=True)
            print(mic_simulated_dir)
            sf.write(mic_simulated_dir, fg_signals[mic_idx], args.sr)
            print(f'Saved {mic_simulated_dir}')

    parent_dir = Path(args.output_path).parent
    room_path = Path(parent_dir, 'RoomPlots')
    mic_dim = transform_mic_coordinates(mic_dim)
    plot_room_new(room,
                  source_coordinates=[voice_loc],  # Since we are processing one voice at a time
                  mic_coordinates=mic_dim,
                  xlim=args.room_dimensions[0],
                  ylim=args.room_dimensions[1],
                  zlim=args.room_dimensions[2],
                  save_path=room_path)
    print('Room plot is saved')


def generate_sample_vox_(args, folder, idx: int) -> str:
    #output_prefix_dir = os.path.join(args.output_path, '{:05d}'.format(idx))
    print(folder)
    folder = os.path.join(args.source_dir, folder)
    print(folder)


    voices_data, video_paths, voice_durations = get_voices(args, folder)
    print(len(video_paths))
    print(video_paths)


    voice_combination_id, last_three_parts = generate_voice_combination_id_vox(video_paths)
    print(f'The voice combination ID is: {voice_combination_id}')
    print(f'The length of voices_data is {len(voices_data)}')

    all_fg_signals = []
    all_voice_positions = []
    e_absorption, max_order = pra.inverse_sabine(args.rt60, args.room_dimensions)
    if args.max_order is not None:
        max_order = args.max_order
    materials = pra.Material(e_absorption)

    for voice_idx, (voice, duration) in enumerate(zip(voices_data, voice_durations)):
        print(f'---------------------------------------\n'
              f'Duration of the audio is: {duration}\n'
              f'---------------------------------------')
        voice_idx = 1

        room = create_room(args.sr,
                           shoebox=True,
                           room_dimensions=args.room_dimensions,
                           absorption_coefficient=e_absorption,
                           max_order=max_order,
                           materials=materials,
                           ray_tracing=True,
                           air_absorption=True)
        mic_center = [x * 0.5 for x in args.room_dimensions[:2]]
        mic_center.append(args.avg_human_height)
        print(f'The mic center coordinates are: {mic_center}')
        total_samples = int(duration * args.sr)
        mic_dim = add_circular_array(room, mic_center, args.mic_radius, args.n_mics)
        max_radius = min(args.room_dimensions[0], args.room_dimensions[1]) / 2

        if not args.random:
            # Use voice_identity to adjust the angle
            voice_theta = np.radians(args.angle_between_sources * args.voice_identity)
            voice_radius = max_radius / 2
        else:
            voice_theta = np.random.uniform(low=0, high=2 * np.pi)
            voice_radius = np.random.uniform(low=0, high=max_radius)

        voice_loc = [
            voice_radius * np.cos(voice_theta) + mic_center[0],
            voice_radius * np.sin(voice_theta) + mic_center[1],
            args.avg_human_height
        ]

        voice_loc = [
            max(0, min(voice_loc[0], args.room_dimensions[0])),
            max(0, min(voice_loc[1], args.room_dimensions[1])),
            args.avg_human_height
        ]

        all_voice_positions.append(voice_loc)
        print(f'The voice has this location: {voice_loc}')
        room.add_source(voice_loc, signal=voices_data[voice_idx][0])

        room.image_source_model()
        room.simulate()
        fg_signals = room.mic_array.signals[:, :total_samples]

        fg_target = np.random.uniform(FG_VOL_MIN, FG_VOL_MAX)
        fg_signals = fg_signals * fg_target / abs(fg_signals).max()
        all_fg_signals.append(fg_signals)

    simulated_RIR_dir = Path(args.output_path)
    simulated_RIR_dir.mkdir(parents=True, exist_ok=True)
    all_signals = []

    for mic_idx in range(args.n_mics):
        mic_simulated_dir = os.path.join(simulated_RIR_dir, "_".join(last_three_parts))
        mic_simulated_dir = Path(mic_simulated_dir)
        print('--------\n--------\n--------\n--------\n--------')
        print(last_three_parts)
        print('--------\n--------\n--------\n--------\n--------')

        for voice_idx, signal in enumerate(all_fg_signals):
            voice_idx = args.voice_identity
            base_name = last_three_parts[0][:-4] if voice_idx == 0 else last_three_parts[1][:-4]
            filename = f"{base_name}_mic{mic_idx}_voice{voice_idx}.wav"
            mic_simulated_dir = simulated_RIR_dir / filename
            mic_simulated_dir = Path(mic_simulated_dir)
            print(f'mic_simulated_dir: {os.path.dirname(mic_simulated_dir)}')
            print('--------\n--------\n--------\n--------\n--------')
            mic_simulated_dir.parent.mkdir(parents=True, exist_ok=True)
            sf.write(mic_simulated_dir, signal[mic_idx], args.sr)
            print(mic_simulated_dir)


        all_signals.append(signal[mic_idx])
        print(len(all_signals))
        print(signal[mic_idx].shape)
        multi_channel_array = np.stack(all_signals, axis=0)
        print(multi_channel_array.shape)
        multi_filename = f'multi_channel_.wav'
        multi_path = os.path.join(simulated_RIR_dir, multi_filename)
        sf.write(multi_path, multi_channel_array.T, args.sr)

        mixed_filename = f"{voice_combination_id}_mic{mic_idx}.wav"
        #parent_dir = Path(args.output_path).parent
        #json_dir = Path(os.path.join(parent_dir, 'metadata'))
        #print(json_dir)

        #json_dir.mkdir(parents=True, exist_ok=True)
        #mixed_filename = mixed_filename.replace(".wav", ".json")
        #file_path = os.path.join(json_dir, mixed_filename)
        #with open(Path(file_path), 'w') as f:
         #   json.dump({
          #      "mic_idx": mic_idx,
           #     "fg_vol": fg_target,
            #    "voice_positions": voice_loc,
             #   "absorption": e_absorption,
            #}, f)

    parent_dir = Path(args.output_path).parent
    room_path = Path(parent_dir, 'RoomPlots')
    mic_dim = transform_mic_coordinates(mic_dim)
    print(mic_dim)
    plot_room_new(room,
              source_coordinates=all_voice_positions,
              mic_coordinates=mic_dim,
              xlim=args.room_dimensions[0],
              ylim=args.room_dimensions[1],
              zlim=args.room_dimensions[2],
              save_path=room_path)
    print('Room is saved')