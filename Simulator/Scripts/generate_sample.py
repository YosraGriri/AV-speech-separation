import argparse
import soundfile as sf
from simulate_room import *
from plot import *
from get_voice import get_voices
import json
from collections import defaultdict
from utils import *

# Mean and STD of the signal peak
FG_VOL_MIN = 0.15
FG_VOL_MAX = 0.4
BG_VOL_MIN = 0.2
BG_VOL_MAX = 0.5


def generate_unique_samples(args, num_samples):
    """
    Generate multiple unique samples with simulated room acoustics without repeating
    the same mix of WAV files.

    Parameters:
    - args: argparse.Namespace, Command-line arguments and parameters for sample generation.
    - num_samples: int, Number of unique samples to generate.

    This function maintains a record of used voice combinations to ensure no repetition.
    """
    used_combinations = defaultdict(bool)  # Tracks used combinations of voice files

    for idx in range(num_samples):
        unique_found = False
        attempt = 0

        while not unique_found:
            # Generate a sample
            voice_combination, output_info = generate_sample_vox(args, idx)
            print(f'The voice combi: {voice_combination}')
            # Check if the combination has been used
            if not used_combinations[voice_combination]:
                used_combinations[voice_combination] = True
                unique_found = True
                print(f"Generated unique sample {idx} with voices: {voice_combination}")
            else:
                attempt += 1
                print(f"Attempt {attempt}: Combination {voice_combination} was already used. Retrying...")

            if attempt > 10:  # Avoid infinite loops if running out of unique combinations
                print("Maximum attempts reached. Unable to find a new unique combination.")
                break

        if not unique_found:
            break  # Exit if we cannot find a unique combination after several attempts

    print("Finished generating unique samples.")


def generate_sample_vox(args: argparse.Namespace, idx: int) -> str:
    output_prefix_dir = os.path.join(args.output_path, '{:05d}'.format(idx))
    #Path(output_prefix_dir).mkdir(parents=True, exist_ok=True)
    voices_data, video_paths, voice_durations = get_voices(args)

    # Use the updated function to generate combination ID and modified paths
    voice_combination_id, last_three_parts = generate_voice_combination_id_vox(video_paths)
    print(f'the last three parts are: {last_three_parts}')

    print(f'The voice combination ID is: {voice_combination_id}')
    print(f'the length of voices_data is{len(voices_data)}')

    # [3] Configure the room and add sound sources
    all_fg_signals = []
    all_voice_positions = []  # Store coordinates of all sources
    e_absorption, max_order = pra.inverse_sabine(args.rt60, args.room_dimensions)
    # Check if args.max_order is not None and update max_order accordingly
    if args.max_order is not None:
        max_order = args.max_order
    materials = pra.Material(e_absorption)

    for voice_idx, (voice, duration) in enumerate(zip(voices_data, voice_durations)):



        print(f'---------------------------------------\n'
              f'duration of the audio is: {duration}\n'
              f'---------------------------------------')

        print(f'Length of voices_data: {len(voices_data)}, Current voice_idx: {voice_idx}')

        print(f'Index voice: {voice_idx}')

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
        total_samples = int(duration * args.sr)
        mic_dim = add_circular_array(room, mic_center, args.mic_radius, args.n_mics)
        max_radius = min(args.room_dimensions[0], args.room_dimensions[1]) / 2

        # Check if the angle and radius should be set to fixed values or chosen randomly
        # Generate a random angle for the speech source within the range [0, 2π]

        if not args.random:
            # Set the angle and radius to the user-provided fixed values
            # Ensure that args.source_angle is given in radians if it's consistent with the random generation range of [0, 2π]
            voice_theta = np.radians(args.angle_between_sources)
            print(f'The voice theta is: {voice_theta}')
            voice_radius = (args.room_dimensions[0] + args.room_dimensions[1])/2
        else:
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
        print('----------------------------------')
        room.add_source(voice_loc, signal=voices_data[voice_idx][0])

        room.image_source_model()
        room.simulate()
        fg_signals = room.mic_array.signals[:, :total_samples]

        # Volume of the sources
        fg_target = np.random.uniform(FG_VOL_MIN, FG_VOL_MAX)
        fg_signals = fg_signals * fg_target / abs(fg_signals).max()
        all_fg_signals.append(fg_signals)

    simulated_RIR_dir = Path(args.output_path)
    simulated_RIR_dir.mkdir(parents=True, exist_ok=True) # Ensure the base directory exists
    all_signals = []

    for mic_idx in range(args.n_mics):
        # Modify the path for saving individual simulations according to the new structure

        mic_simulated_dir = os.path.join(simulated_RIR_dir, "_".join(last_three_parts))
        mic_simulated_dir = Path(mic_simulated_dir)
        print('--------\n--------\n--------\n--------\n--------')
        print(last_three_parts)
        print('--------\n--------\n--------\n--------\n--------')

        for voice_idx, signal in enumerate(all_fg_signals):
            # Adjusting the filename construction based on voice_idx
            if voice_idx == 0:
                base_name = last_three_parts[0][:-4]
            else:  # voice_idx == 1, using last_three_parts[1][-4]
                base_name = last_three_parts[1][:-4]

            # Constructing filename for each voice
            filename = f"{base_name}_mic{mic_idx}_voice{voice_idx}.wav"

            # Constructing the full path for the file to be saved
            mic_simulated_dir = simulated_RIR_dir / filename
            mic_simulated_dir = Path(mic_simulated_dir)
            print(mic_simulated_dir)
            print('--------\n--------\n--------\n--------\n--------')


            print('--------\n--------\n--------\n--------\n--------')
           #### A comment for Myself, Path Creation here
            # Ensuring the parent directory of the file exists
            mic_simulated_dir.parent.mkdir(parents=True, exist_ok=True)
            # Saving the signal to file
            sf.write(mic_simulated_dir, signal[mic_idx], args.sr)
        all_signals.append(signal[mic_idx])
        print(len(all_signals))
        print(signal[mic_idx].shape)
        ##### For now not saving mixtures deactivated! #####
        multi_channel_array = np.stack(all_signals, axis=0)
        print(multi_channel_array.shape)
        multi_filename = f'multi_channel_.wav'
        multi_path = os.path.join(simulated_RIR_dir, multi_filename)
        sf.write(multi_path, multi_channel_array.T, args.sr)


        # Combine and save the mixed signal in the designated 'mixed' directory
        # mixed_dir = os.path.join(simulated_RIR_dir, "mixed")
        # Path(mixed_dir).mkdir(parents=True, exist_ok=True)
        mixed_filename = f"{voice_combination_id}_mic{mic_idx}.wav"
        # mixed_path = os.path.join(mixed_dir, mixed_filename)
        # mixed_path = Path(mixed_path)
        # print(mixed_path)
        # print('--------\n--------\n--------\n--------\n--------')
        # mixed_signal = sum(all_fg_signals)  # This is a simple sum of signals; adjust according to your needs
        # sf.write(mixed_path, mixed_signal, args.sr)

        # Save metadata
        parent_dir = Path(args.output_path).parent
        json_dir = Path(os.path.join(parent_dir, 'metadata'))
        print(json_dir)

        json_dir.mkdir(parents=True, exist_ok=True)
        mixed_filename = mixed_filename.replace(".wav", ".json")
        file_path = os.path.join(json_dir, mixed_filename)
        with open(Path(file_path), 'w') as f:
            json.dump({
                "mic_idx": mic_idx,
                "fg_vol": fg_target,
                "voice_positions": voice_loc,
                "absorption": e_absorption,
            }, f)
        # print(f' the json file is saved')

    # [6]
    parent_dir = Path(args.output_path).parent
    room_path = Path(parent_dir, 'RoomPlots')
    plot_room(room,
              source_coordinates=all_voice_positions,
              mic_coordinates=mic_dim,
              xlim=args.room_dimensions[0],
              ylim=args.room_dimensions[1],
              zlim=args.room_dimensions[2],
              save_path=room_path)
    print('room is saved')

    # Copy video files to the output directory
    #for video_path in video_paths:
     #   shutil.copy2(video_path, output_prefix_dir)
    metadata = {
        "output_prefix_dir": output_prefix_dir,
        "voice_combination_id": voice_combination_id,
        # Include other metadata as needed
    }
    return (voice_combination_id,
            {"output_prefix_dir": output_prefix_dir,
             "voice_combination_id": voice_combination_id})