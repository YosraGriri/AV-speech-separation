import librosa
from pathlib import Path
import json
import argparse


def find_wav_files_recursively(root_dir):
    """
    Searches recursively for all .wav files within the given root directory and its subdirectories.
    Args:
        root_dir (str): The path to the root directory from where the search for .wav files will begin.
    Returns:
        list[str]: A list of the paths to the .wav files found, represented as strings.
    """
    wav_files = []
    source_dirs = [d for d in Path(root_dir).iterdir() if d.is_dir()]
    for source_dir in source_dirs:
        wav_files.extend(list(source_dir.rglob('*.wav')))
    return [str(file) for file in wav_files]


def collect_voice_metadata(args):
    """
    Loops over a list selects non-silent voices from the specified directory, delving into subfolders as needed.
    Parameters:
        args (Namespace): Command-line arguments or configuration parameters. Expected to include 'sr' for sampling rate.
        root_dir (str): The path to the root directory to search within.
    Returns:
        voices_data (list of tuples): Each tuple contains voice data (numpy array) and its identity.
        wav_paths (list of str): Paths to voice files found.
        voice_durations (list of float): Duration of each voice file in seconds.
    """
    voices_data, wav_paths, voice_durations = [], [], []
    wav_files = find_wav_files_recursively(args.source_dir)

    for wav_file in wav_files:
        if wav_file not in wav_paths:
            wav_paths.append(wav_file)
            print('tada')

        voice_identity = Path(wav_file).stem.split("_")[0]
        voice, sr = librosa.load(wav_file, sr=args.sr, mono=True)
        duration = len(voice) / sr
        voice_durations.append(duration)

        if voice.std() == 0:  # Skip silent voices
            continue

        voices_data.append((voice, voice_identity))
        #voices_data = voices_data[:n_files]
        #wav_paths = wav_paths [:n_files]
        #voice_durations = voice_durations[:n_files]
    return voices_data, wav_paths, voice_durations

def save_metadata_to_file(voices_data, wav_paths, voice_durations, output_file_path):
    """
    Saves the metadata (voice data, wav paths, voice durations) to a JSON file.

    Args:
        voices_data (list): A list of tuples containing voice data and identity.
        wav_paths (list): A list of paths to voice files.
        voice_durations (list): A list of durations for each voice file.
        output_file_path (str or Path): The path to the output file where data will be saved.
    """
    # Prepare the data to be saved. Note: voice_data is not saved directly due to its potentially large size and complexity.
    data_to_save = {
        "voice_durations": voice_durations,
        "wav_paths": wav_paths
    }

    # Write the data to a JSON file
    with open(output_file_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)


def main_collect_voice_metadata(args):
    voices_data, wav_paths, voice_durations = collect_voice_metadata(args)

    # Define the output file path
    parent_dir = Path(args.source_dir).parent
    output_file_path = parent_dir / "voice_metadata.json"

    # Save the metadata to a file
    save_metadata_to_file(voices_data, wav_paths, voice_durations, output_file_path)

    print(f"Metadata saved to {output_file_path}")

