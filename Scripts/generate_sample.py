from pathlib import Path
import argparse
import soundfile as sf
from simulate_room import *
from plot import *
from get_voice import get_voices
import json
# Mean and STD of the signal peak
FG_VOL_MIN = 0.15
FG_VOL_MAX = 0.4
BG_VOL_MIN = 0.2
BG_VOL_MAX = 0.5


def generate_sample(args: argparse.Namespace, idx: int) -> int:
    """
    Generate a single sample with simulated room acoustics.

    Parameters:
    - args: argparse.Namespace, Command-line arguments and parameters.
    - room: pyroomacoustics Room object, representing the simulated room.
    - idx: int, Index of the generated sample.

    Returns:
    - int: 0 on success.

    Steps:
    - [1] Load voice signals.
    - [2] Create an output directory for the sample.
    - [3] Sample background with the same length as voice signals.
    - [4] Configure the room and add sound sources.
    - [5] Simulate room acoustics and render sound signals.
    - [6] Save audio files for individual voices, mixed audio, and metadata.

    Note: This function simulates a room with multiple sound sources and records the resulting signals
    at multiple microphones. It saves individual audio files for each voice, a mixed audio file,
    and metadata describing the simulation.

    """

    # [1] Load voice signals
    global room, mic_dim, fg_target
    output_prefix_dir = os.path.join(args.source_dir, '{:05d}'.format(idx))
    Path(output_prefix_dir).mkdir(parents=True, exist_ok=True)
    voices_data = get_voices(args)
    print(f'the length of voices_data is{len(voices_data)}')

    # [3] Sample background with the same length as voice signals
    total_samples = int(args.duration * args.sr)

    # [4] Configure the room and add sound sources
    all_fg_signals = []
    all_voice_positions = []  # Store coordinates of all sources
    e_absorption, max_order = pra.inverse_sabine(args.rt60, args.room_dimensions)
    # Check if args.max_order is not None and update max_order accordingly
    if args.max_order is not None:
        max_order = args.max_order
    materials = pra.Material(e_absorption)

    for voice_idx in range(args.n_sources):
        print(f'index voice: {voice_idx}')
        # Re-generate room to save ground truth
        room = create_room(args.sr,
                           shoebox=True,
                           room_dimensions=args.room_dimensions,
                           absorption_coefficient=e_absorption,
                           max_order=max_order,
                           materials=materials,
                           ray_tracing=True,
                           air_absorption=True)
        ##Adding mic array
        mic_center = [x * 0.5 for x in args.room_dimensions[:2]]
        mic_center.append(args.avg_human_height)
        print(f'The mic center coordinates are:{mic_center}')

        mic_dim = add_circular_array(room, mic_center, args.mic_radius, args.n_mics)

        # Generate a random 3D location for the speech source.
        # Angle 'voice_theta' determines direction, 'voice_radius' sets distance,
        # and 'args.avg_human_height' defines a fixed height.
        # Calculate the maximum radius allowed based on the room dimensions
        # Generate a random angle between voice sources
        # Calculate the maximum radius allowed based on the room dimensions
        max_radius = min(args.room_dimensions[0], args.room_dimensions[1]) / 2

        # Generate a random angle for the speech source within the range [0, 2π]
        voice_theta = np.random.uniform(low=0, high=2 * np.pi)

        # Generate a random radius within the range [0, max_radius]
        voice_radius = np.random.uniform(low=0, high=max_radius)

        # Calculate the corresponding coordinates based on the generated angle and radius
        voice_loc = [
            voice_radius * np.cos(voice_theta) + mic_center[0],
            voice_radius * np.sin(voice_theta) + mic_center[1],
            args.avg_human_height
        ]

        # Ensure that the generated coordinates are within the room boundaries by clamping them
        voice_loc = [
            max(0, min(voice_loc[0], args.room_dimensions[0])),
            max(0, min(voice_loc[1], args.room_dimensions[1])),
            args.avg_human_height
        ]

        all_voice_positions.append(voice_loc)  # Append coordinates of current source to the list

        print(f'The voice has this location {voice_loc}')
        print(voice_idx)
        print(voices_data)
        room.add_source(voice_loc, signal=voices_data[voice_idx][0])

        room.image_source_model()
        room.simulate()
        fg_signals = room.mic_array.signals[:, :total_samples]

        # Volume of the sources
        fg_target = np.random.uniform(FG_VOL_MIN, FG_VOL_MAX)
        fg_signals = fg_signals * fg_target / abs(fg_signals).max()
        all_fg_signals.append(fg_signals)

    # [6] Save audio files for individual voices, mixed audio, and metadata
    for mic_idx in range(args.n_mics):
        output_prefix = str(
            Path(output_prefix_dir) / "mic{:02d}_".format(mic_idx))

        # Save individual voice signals
        all_fg_buffer = np.zeros((total_samples))
        for voice_idx in range(args.n_sources):
            curr_fg_buffer = np.pad(all_fg_signals[voice_idx][mic_idx],
                                    (0, total_samples))[:total_samples]
            sf.write(output_prefix + "voice{:02d}.wav".format(voice_idx),
                     curr_fg_buffer, args.sr)
            all_fg_buffer += curr_fg_buffer

        # Save mixed audio signal
        mixed_buffer = all_fg_buffer
        output_filename = output_prefix + "mixed.wav"
        sf.write(output_filename, mixed_buffer, args.sr)

        # Save metadata
        with open(output_filename.replace(".wav", ".json"), 'w') as f:
            json.dump({
                "mic_idx": mic_idx,
                "fg_vol": fg_target,
                "voice_positions": voice_loc,
                "absorption": e_absorption,
            }, f)
        print(f' the json file is saved')

    # [6]
    plot_room(room,
              source_coordinates=all_voice_positions,
              mic_coordinates=mic_dim,
              xlim=args.room_dimensions[0],
              ylim=args.room_dimensions[1],
              zlim=args.room_dimensions[2],
              save_path=args.output_path)
    print('room is saved')
    return 0
