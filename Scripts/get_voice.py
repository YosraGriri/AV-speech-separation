import os
import random
import librosa
from pathlib import Path


# Function to recursively find a folder with wav files
def find_wav_files(folder):
    """
    Navigates through a nested directory structure to find .wav audio files.

    Starting from the specified 'folder', it randomly selects subdirectories
    to continue the search if subdirectories exist. This process is repeated
    until a directory without further subdirectories is reached. At this point,
    it searches for and returns a list of .wav files found within. If no .wav
    files are present in the final directory, an empty list is returned.
    Parameters:
    - folder (str): The starting directory path for the search.
    Returns:
    - list: A list of paths to .wav files found in the directory without
            further subdirectories. Returns an empty list if no .wav files
            are found.
    """
    while True:
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        if not subfolders:
            # No subfolders, check for wav files in the current folder
            wav_files = [f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith('.wav')]
            return wav_files
        else:
            # Randomly select a subfolder to dive into
            folder = random.choice(subfolders)

def get_voices(args):
    """
    Randomly selects non-silent voices from the specified directory, delving into subfolders as needed.

    Parameters:
    - args: Namespace, Command-line arguments or configuration parameters.

    Returns:
    - voices_data: list of tuples, Each tuple contains voice data (numpy array) and its identity.
    - video_paths: list of str, Paths to voice files found.
    """
    voices_data = []
    video_paths = []
    selected_wav_files = find_wav_files(args.source_dir)

    for selected_wav_file in selected_wav_files:
        if selected_wav_file not in video_paths:
            video_paths.append(selected_wav_file)

        voice_identity = Path(selected_wav_file).stem.split("_")[0]
        voice, _ = librosa.load(selected_wav_file, sr=args.sr, mono=True)
        voice, _ = librosa.effects.trim(voice)

        if voice.std() == 0:  # Skip silent voices
            continue

        voices_data.append((voice, voice_identity))

        if len(voices_data) >= args.n_sources:
            break

    return voices_data, video_paths
