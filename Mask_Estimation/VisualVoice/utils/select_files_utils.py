import os
import random
from pathlib import Path


def select_random_files(base_dir, n_mic, voice_id):
    # Define paths to the main categories
    mouth_roi_base = os.path.join(base_dir, 'mouth_roi/unseen_unheard_test')
    audio_base = os.path.join(base_dir, 'raw_audio_simulated')
    video_base = os.path.join(base_dir, 'mp4')

    # Select two random Parent IDs from the mouth_roi directory
    parent_ids = random.sample(os.listdir(mouth_roi_base), 2)
    selections = []

    for index, parent_id in enumerate(parent_ids):
        # Path to the specific Parent ID directory under mouth_roi
        mouth_roi_parent_path = os.path.join(mouth_roi_base, parent_id)
        # Select a random ID folder from this directory
        id_folder = random.choice(os.listdir(mouth_roi_parent_path))

        # Construct paths for mouth_roi, audio, and video for this ID
        mouth_roi_path = os.path.join(mouth_roi_parent_path, id_folder)
        audio_path = os.path.join(audio_base, parent_id, id_folder)
        video_path = os.path.join(video_base, parent_id, id_folder)

        # Select a random .h5 file (mouth_roi) and save its ID
        mouth_roi_file = random.choice(os.listdir(mouth_roi_path))
        file_id = mouth_roi_file.split('.')[0]  # Extract the file ID, like '00039' from '00039.h5'

        # Locate the corresponding video file
        video_file = file_id + '.mp4'  # Assuming the video file follows this naming convention
        video_file_path = os.path.join(video_path, video_file)

        # Generate audio file paths according to the naming convention
        audio_file_full_paths = [os.path.join(audio_path, f"{file_id}_mic{i}_voice{str(voice_id)}.wav") for i in range(n_mic)]

        # For the first parent ID, assign to audio1, for the second, assign to audio2
        if index == 0:
            audio_file_full_paths = [os.path.join(audio_path, f"{file_id}_mic{i}_voice0.wav") for i in
                                     range(n_mic)]
            audio1_paths = audio_file_full_paths
            mouth_roi1_path = os.path.join(mouth_roi_path, mouth_roi_file)
            video1_path = video_file_path
        else:
            audio_file_full_paths = [os.path.join(audio_path, f"{file_id}_mic{i}_voice1.wav") for i in
                                     range(n_mic)]
            audio2_paths = audio_file_full_paths
            mouth_roi2_path = os.path.join(mouth_roi_path, mouth_roi_file)
            video2_path = video_file_path

    # Now pair up each microphone file from audio1_paths with a corresponding one from audio2_paths
    for mic_num in range(n_mic):
        selections.append({
            'audio1_path': str(Path(audio1_paths[mic_num])),
            'audio2_path': str(Path(audio2_paths[mic_num])),
            'mouthroi1_path': str(Path(mouth_roi1_path)),
            'mouthroi2_path': str(Path(mouth_roi2_path)),
            'video1_path': str(Path(video1_path)),
            'video2_path': str(Path(video2_path))
        })

    return selections


def select_random_files_original(base_dir, n_pairs=100):
        # Define paths to the main categories
        mouth_roi_base = os.path.join(base_dir, 'mouth_roi/unseen_unheard_test')
        mouth_roi_base = "E:/AV-speech-separation/data/VoxCeleb2/mouth_roi/unseen_unheard_test"
        audio_base = os.path.join(base_dir)
        video_base = os.path.join(base_dir, 'mp4')

        # Get all random IDs
        all_random_ids = os.listdir(mouth_roi_base)

        selections = []

        if len(all_random_ids) < 2:
            raise ValueError("Not enough unique random ID directories to select pairs")

        for _ in range(n_pairs):
            # Select a random ID for each pair
            selected_random_ids = random.sample(all_random_ids, 2)

            for index, random_id in enumerate(selected_random_ids):
                # Path to the specific random ID directory under mouth_roi
                mouth_roi_random_path = os.path.join(mouth_roi_base, random_id)

                # List all speaker IDs in the current random ID folder
                speaker_ids = os.listdir(mouth_roi_random_path)

                # Ensure there are enough speaker IDs to select
                if len(speaker_ids) < 1:
                    raise ValueError("Not enough unique speaker IDs in the selected random ID folder")

                # Select a random speaker ID
                selected_speaker_id = random.choice(speaker_ids)

                # Path to the specific speaker ID directory under the random ID
                mouth_roi_speaker_path = os.path.join(mouth_roi_base, random_id, selected_speaker_id)
                audio_speaker_path = os.path.join(audio_base, random_id, selected_speaker_id)
                video_speaker_path = os.path.join(video_base, random_id, selected_speaker_id)

                # Select a random segment from this directory
                segment_folder = random.choice(os.listdir(mouth_roi_speaker_path))

                # Construct paths for mouth_roi, audio, and video for this segment
                mouth_roi_path = os.path.join(mouth_roi_speaker_path, segment_folder)
                audio_path = os.path.join(audio_speaker_path, segment_folder)
                video_path = os.path.join(video_speaker_path, segment_folder)

                # Select a random .h5 file (mouth_roi) and save its ID
                mouth_roi_file = random.choice(os.listdir(Path(mouth_roi_path)))
                file_id = mouth_roi_file.split('.')[0]  # Extract the file ID, like '00039' from '00039.h5'

                # Locate the corresponding video file
                video_file = file_id + '.mp4'  # Assuming the video file follows this naming convention
                video_file_path = os.path.join(video_path, video_file)

                # Generate audio file paths
                audio_file_path = os.path.join(audio_path, f"{file_id}.wav")

                # For the first speaker ID, assign to audio1, for the second, assign to audio2
                if index == 0:
                    audio1_path = audio_file_path
                    mouth_roi1_path = os.path.join(mouth_roi_path, mouth_roi_file)
                    video1_path = video_file_path
                else:
                    audio2_path = audio_file_path
                    mouth_roi2_path = os.path.join(mouth_roi_path, mouth_roi_file)
                    video2_path = video_file_path

            # Add the selected paths to the selections list
            selections.append({
                'audio1_path': str(Path(audio1_path)),
                'audio2_path': str(Path(audio2_path)),
                'mouthroi1_path': str(Path(mouth_roi1_path)),
                'mouthroi2_path': str(Path(mouth_roi2_path)),
                'video1_path': str(Path(video1_path)),
                'video2_path': str(Path(video2_path))
            })

        return selections
