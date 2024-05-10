import librosa

# Replace 'path_to_audio_file.wav' with the actual path to one of your audio files
audio_path = "../../data/VoxCeleb2/raw_audio_test/id04030/7mXUMuo5_NE/00001.wav"
audio, sr = librosa.load(audio_path, sr=16000)  # Load with original sample rate
duration = len(audio) / sr
print(f"Duration: {duration} seconds")