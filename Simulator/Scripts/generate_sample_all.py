import soundfile as sf
from simulate_room import *
from plot import *
from get_voice import get_voices
import os
from pathlib import Path
import numpy as np

# Constants defining the minimum and maximum volume levels for foreground and background signals
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
    # Get a list of all directories in the source directory
    folders = [f for f in os.listdir(args.source_dir) if os.path.isdir(os.path.join(args.source_dir, f))]

    # Process only the first two folders for room simulation
    for idx, folder in enumerate(folders[0:2]):
        args.folder = folder
        print(f"Generating room simulation for folder: {folder}")
        generate_sample_vox(args, folder, idx)  # Use folder index for unique identification


def generate_sample_vox(args, folder, idx: int) -> str:
    """
    Generate a sample voice simulation for a given folder.

    Parameters:
    - args: argparse.Namespace, Command-line arguments and parameters for sample generation.
    - folder: str, The folder containing voice samples.
    - idx: int, The index of the folder being processed.
    """
    folder_path = os.path.join(args.source_dir, folder)

    # Retrieve voice data, video paths, and voice durations from the folder
    voices_data, video_paths, voice_durations = get_voices(args, folder_path)
    print(f'Number of video paths: {len(video_paths)}')
    print(f'Video paths: {video_paths}')

    # Calculate the room's absorption coefficient and maximum order for the room simulation
    e_absorption, max_order = pra.inverse_sabine(args.rt60, args.room_dimensions)
    if args.max_order is not None:
        max_order = args.max_order
    materials = pra.Material(e_absorption)

    for voice_idx, (voice, duration) in enumerate(zip(voices_data, voice_durations)):
        print(f'Processing voice {voice_idx} with duration {duration}s')

        video_path = video_paths[voice_idx]
        # Extract the actual voice signal assuming it's the first element
        voice_signal = voice[0] if isinstance(voice, (list, tuple)) else voice

        # Ensure the voice signal is a numpy array and has the correct shape
        voice_signal = np.array(voice_signal)
        if len(voice_signal.shape) != 1:
            raise ValueError(f"Expected a 1D array for the voice signal, got shape {voice_signal.shape}")

        # Create the room with specified parameters
        room = create_room(args.sr,
                           shoebox=True,
                           room_dimensions=args.room_dimensions,
                           absorption_coefficient=e_absorption,
                           max_order=max_order,
                           materials=materials,
                           ray_tracing=True,
                           air_absorption=True)

        # Calculate the microphone center coordinates
        mic_center = [x * 0.5 for x in args.room_dimensions[:2]]
        mic_center.append(args.avg_human_height)
        print(f'The mic center coordinates are: {mic_center}')

        # Determine the total number of samples for the voice signal
        total_samples = int(duration * args.sr)
        mic_dim = add_circular_array(room, mic_center, args.mic_radius, args.n_mics)
        max_radius = min(args.room_dimensions[0], args.room_dimensions[1]) / 2

        # Determine the location of the voice source
        voice_theta = np.random.uniform(low=0, high=2 * np.pi) if args.random else np.radians(
            args.angle_between_sources * voice_idx)
        voice_radius = np.random.uniform(low=0, high=max_radius) if args.random else max_radius / 2

        voice_loc = [
            voice_radius * np.cos(voice_theta) + mic_center[0],
            voice_radius * np.sin(voice_theta) + mic_center[1],
            args.avg_human_height
        ]

        # Ensure the voice location is within room boundaries
        voice_loc = [
            max(0, min(voice_loc[0], args.room_dimensions[0])),
            max(0, min(voice_loc[1], args.room_dimensions[1])),
            args.avg_human_height
        ]

        print(f'The voice has this location: {voice_loc}')

        # Add the voice source to the room
        room.add_source(voice_loc, signal=voice_signal)

        # Simulate the room acoustics
        room.image_source_model()
        room.simulate()

        # Get the microphone array signals
        fg_signals = room.mic_array.signals[:, :total_samples]

        # Adjust the foreground signal volume
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

    # Save the room plot
    parent_dir = Path(args.output_path).parent
    room_path = Path(parent_dir, 'RoomPlots')
    mic_dim = transform_mic_coordinates(mic_dim)
    plot_room(room,
              source_coordinates=[voice_loc],  # Since we are processing one voice at a time
              mic_coordinates=mic_dim,
              xlim=args.room_dimensions[0],
              ylim=args.room_dimensions[1],
              zlim=args.room_dimensions[2],
              save_path=room_path)
    print('Room plot is saved')

