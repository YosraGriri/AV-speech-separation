import argparse
import soundfile as sf
from simulate_room import *
from plot import *
from collect_voice_metadata import collect_voice_metadata
import json
from utils import *

# Mean and STD of the signal peak
FG_VOL_MIN = 0.15
FG_VOL_MAX = 0.4
BG_VOL_MIN = 0.2
BG_VOL_MAX = 0.5


def create_room_simulations(args, n_samples=None):
    global filename, fg_target, voice_loc, mic_dim
    voices_data, wav_paths, voice_durations = collect_voice_metadata(args)
    if n_samples is not None:
        voices_data = voices_data[:n_samples]
        wav_paths = wav_paths[:n_samples]
        voice_durations = voice_durations[:n_samples]

    last_three_parts = [os.path.join(*Path(path).parts[-3:]) for path in wav_paths]# Get the last three par
    # [3] Configure the room and add sound sources
    all_fg_signals = []
    all_voice_positions = []  # Store coordinates of all sources
    if args.rt60 > 0.0:
        e_absorption, max_order = pra.inverse_sabine(args.rt60, args.room_dimensions)
    else: #Anechoic chamber
        e_absorption = 1.0
    # Check if args.max_order is not None and update max_order accordingly
    if args.max_order is not None:
        max_order = args.max_order
    materials = pra.Material(e_absorption)

    for voice_idx, (voice, duration) in enumerate(zip(voices_data, voice_durations)):
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
        if not args.random:
            # Set the angle and radius to the user-provided fixed values
            # Ensure that args.source_angle is given in radians if it's consistent with the random generation range of [0, 2π]
            voice_theta = np.radians(args.voice_theta)
            voice_radius = args.voice_radius
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
        print(voice_loc)
        # Check if the generated coordinates are outside the room boundaries
        if (voice_loc[0] < 0 or voice_loc[0] > args.room_dimensions[0] or
                voice_loc[1] < 0 or voice_loc[1] > args.room_dimensions[1]):
            raise ValueError("Calculated voice location is outside the room boundaries.")

        all_voice_positions.append(voice_loc)  # Append coordinates of current source to the list
        print(f'The voice has this location {voice_loc}')
        print('--------\n--------')
        room.add_source(voice_loc, signal=voices_data[voice_idx][0])
        room.image_source_model()
        room.simulate()
        fg_signals = room.mic_array.signals[:, :total_samples]

        # Volume of the sources
        fg_target = (FG_VOL_MIN + FG_VOL_MAX) * 0.5
        #fg_target = np.random.uniform(FG_VOL_MIN, FG_VOL_MAX)
        fg_signals = fg_signals * fg_target / abs(fg_signals).max()
        all_fg_signals.append(fg_signals)

    simulated_rir_dir = Path(args.output_path)
    simulated_rir_dir.mkdir(parents=True, exist_ok=True)  # Ensure the base directory exists

    for mic_idx in range(args.n_mics):
        # Modify the path for saving individual simulations according to the new structure

        mic_simulated_dir = os.path.join(simulated_rir_dir, "_".join(last_three_parts))
        mic_simulated_dir = Path(mic_simulated_dir)
        print(mic_simulated_dir)
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
            mic_simulated_dir = simulated_rir_dir / filename
            mic_simulated_dir = Path(mic_simulated_dir)
            print(f'the directory of the simulated file is {mic_simulated_dir}:'
                  f'\n--------\n--------\n--------')
            #### A comment for Myself, Path Creation here
            # Ensuring the parent directory of the file exists
            mic_simulated_dir.parent.mkdir(parents=True, exist_ok=True)
            # Saving the signal to file
            sf.write(mic_simulated_dir, signal[mic_idx], args.sr)

        # Save metadata
        parent_dir = Path(args.output_path).parent
        json_dir = Path(os.path.join(parent_dir, 'metadata'))
        print(json_dir)

        json_dir.mkdir(parents=True, exist_ok=True)

        filename = Path(filename).name.replace(".wav", ".json")
        file_path = os.path.join(json_dir, filename)
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
        return 'room is saved'
