import librosa
from pathlib import Path

def select_all_wav_from_sources(root_dir):
    """
    Selects all .wav files from all source directories under the given root directory.

    Parameters:
    - root_dir (str): The path to the root directory containing source subfolders.

    Returns:
    - list: Paths to all .wav files from different sources.
    """
    source_dirs = [d for d in Path(root_dir).iterdir() if d.is_dir()]
    selected_files = []

    for source_dir in source_dirs:  # Loop through all source directories
        wav_files = list(source_dir.rglob('*.wav'))  # Search for .wav files recursively
        if wav_files:
            selected_files.extend(wav_files)  # Add all .wav files to the list

    return [str(file) for file in selected_files]

def get_voices(args, source_dir):
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
    voice_durations = []
    selected_wav_files = select_all_wav_from_sources(source_dir)
    #print(selected_wav_files)

    for selected_wav_file in selected_wav_files:
        # Convert to a Path object, which automatically handles path normalization
        selected_wav_file = Path(selected_wav_file)
        # Convert back to a string if needed (this will use the correct separator for the OS)
        selected_wav_file = str(selected_wav_file)
        #print(selected_wav_file)
        if selected_wav_file not in video_paths:
            video_paths.append(selected_wav_file)

        voice_identity = args.voice_identity  # Ensure voice_identity is fixed from args
        voice, sr = librosa.load(selected_wav_file, sr=args.sr, mono=True)
        duration = len(voice) / sr
        voice_durations.append(duration)
        if voice.std() == 0:  # Skip silent voices
            continue
        voices_data.append((voice, voice_identity))
    return voices_data, video_paths, voice_durations

