import random
import os
import librosa


import os
import random
import librosa

def get_voices(args):
    """
    Randomly selects non-silent voices from the specified directory or its subfolders.

    Parameters:
    - args: Namespace, Command-line arguments or configuration parameters. Expected to have source_dir, sr, n_sources.

    Returns:
    - voices_data: list of tuples, Each tuple contains voice data (numpy array) and its identity.
    - video_paths: list of str, Paths to video files found in the selected directories.
    """
    # Check if there are subfolders in the source directory
    subfolders = [f.path for f in os.scandir(args.source_dir) if f.is_dir()]

    if len(subfolders) < 1:
        # No subfolders, use the main directory
        selected_folders = [args.source_dir]
    else:
        # Select random subfolders
        selected_folders = random.sample(subfolders, min(len(subfolders), args.n_sources))

    voices_data = []
    video_paths = []

    for folder in selected_folders:
        # Get a list of wav files in the folder
        wav_files = [f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith('.wav')]

        # Skip folders with no WAV files
        if not wav_files:
            continue

        # If less than 1 subfolder is present, select two random wav files from the main directory
        if len(subfolders) < 1 and len(wav_files) > 1:
            selected_wav_files = random.sample(wav_files, 2)  # Ensure there are at least two to choose from
        else:
            selected_wav_files = [random.choice(wav_files)]

        for selected_wav_file in selected_wav_files:
            print(f'We select this file {selected_wav_file}')
            # Extract voice identity from the filename
            voice_identity = os.path.basename(selected_wav_file).split("_")[0]

            # Load and process the voice data
            voice, _ = librosa.load(selected_wav_file, sr=args.sr, mono=True)
            voice, _ = librosa.effects.trim(voice)

            # Skip silent voices
            if voice.std() == 0:
                continue

            # Append the voice data and identity to the list
            voices_data.append((voice, voice_identity))

            # Break if the desired number of voices is reached
            if len(voices_data) >= args.n_sources:
                break

        # This section remains unchanged - it collects video file paths
        video_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp4')]
        video_paths.extend(video_files)

        # If we have collected enough voices, no need to continue
        if len(voices_data) >= args.n_sources:
            break

    return voices_data, video_paths
