import random
import os
import librosa

def get_voices(args):
    """
    Randomly selects non-silent voices from random folders within a specified directory.

    Parameters:
    - directory: str, Path to the main directory containing subfolders with voice data.
    - n_voices_to_take: int, Number of voices to randomly select.
    - n_folders_to_select: int, Number of folders to randomly select.
    - args: Namespace, Command-line arguments or configuration parameters.

    Returns:
    - voices_data: list of tuples, Each tuple contains voice data (numpy array) and its identity.
    """
    # Get a list of subfolders (voice folders)

    subfolders = [f.path for f in os.scandir(args.source_dir) if f.is_dir()]

    # Randomly select n_folders_to_select number of subfolders
    selected_folders = random.sample(subfolders, args.n_sources)

    voices_data = []

    for folder in selected_folders:
        # Get a list of wav files in the selected folder
        wav_files = [f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith('.wav')]

        if not wav_files:
            continue  # Skip folders with no WAV files

        # Randomly select one wav file from the folder
        selected_wav_file = random.choice(wav_files)
        print(f'We select this file {selected_wav_file}')
        # Extract voice identity from the filename
        voice_identity = str(selected_wav_file).split("/")[-1].split("_")[0]

        # Load and process the voice data
        voice, _ = librosa.core.load(selected_wav_file, sr=args.sr, mono=True)
        voice, _ = librosa.effects.trim(voice)

        # Check if the voice is non-silent
        if voice.std() == 0:
            continue  # Skip silent voices

        # Append the voice data and identity to the list
        voices_data.append((voice, voice_identity))

        # Break the loop when the desired number of voices is reached
        if len(voices_data) == args.n_sources:
            break
    return voices_data