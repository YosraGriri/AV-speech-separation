import subprocess
import random
import os
from pprint import pprint
from pathlib import Path

# Base paths
base_dir = "E:/AV-speech-separation/data/VoxCeleb2"
#audio_base_path = Path(os.path.join(base_dir, 'raw_audio_test'))
#video_base_path = Path(os.path.join(base_dir, 'mp4'))
#mouthroi_base_path = Path(os.path.join(base_dir, 'mouth_roi/unseen_unheard'))
output_dir_root = Path(os.path.join(base_dir, 'results'))

# Function to select random files
def select_random_files_original(base_dir, n_pairs=1):
    # Define paths to the main categories
    mouth_roi_base = 'E:/AV-speech-separation/data/VoxCeleb2/mouth_roi/unseen_unheard'
    audio_base = 'E:/AV-speech-separation/data/VoxCeleb2/raw_audio_test'
    video_base = 'E:/AV-speech-separation/data/VoxCeleb2/mp4'

    # Check if the directory exists
    if not os.path.exists(mouth_roi_base):
        print(f"Error: The directory {mouth_roi_base} does not exist.")
        return []

    print(f"The directory {mouth_roi_base} exists.")
    all_random_ids = os.listdir(mouth_roi_base)
    if not all_random_ids:
        print(f"Error: No directories found in {mouth_roi_base}")
        return []

    print(f"Random IDs found: {all_random_ids}")
    selections = []

    if len(all_random_ids) < 2:
        raise ValueError("Not enough unique random ID directories to select pairs")

    for _ in range(n_pairs):
        # Select a random ID for each pair
        selected_random_ids = random.sample(all_random_ids, 2)

        for index, random_id in enumerate(selected_random_ids):
            # Path to the specific random ID directory under mouth_roi
            mouth_roi_random_path = Path(os.path.join(mouth_roi_base, random_id))
            print(f"Checking path: {mouth_roi_random_path}")

            if not mouth_roi_random_path.exists():
                print(f"Error: Directory does not exist: {mouth_roi_random_path}")
                continue

            # List all speaker IDs in the current random ID folder
            speaker_ids = os.listdir(mouth_roi_random_path)
            print(f"Speaker IDs in {mouth_roi_random_path}: {speaker_ids}")

            # Ensure there are enough speaker IDs to select
            if len(speaker_ids) < 1:
                print(f"Error: Not enough unique speaker IDs in the selected random ID folder: {mouth_roi_random_path}")
                continue

            # Select a random speaker ID
            selected_speaker_id = random.choice(speaker_ids)

            # Path to the specific speaker ID directory under the random ID
            mouth_roi_speaker_path = os.path.join(mouth_roi_random_path, selected_speaker_id)
            audio_speaker_path = os.path.join(audio_base, random_id, selected_speaker_id)
            video_speaker_path = os.path.join(video_base, random_id, selected_speaker_id)

            if not os.path.exists(mouth_roi_speaker_path):
                print(f"Error: Directory does not exist: {mouth_roi_speaker_path}")
                continue

            # List all .h5 files in this directory (since segment_folder is a file)
            segment_files = [f for f in os.listdir(mouth_roi_speaker_path) if f.endswith('.h5')]
            print(f"Segment files in {mouth_roi_speaker_path}: {segment_files}")

            # Ensure there are segment files to select from
            if not segment_files:
                print(f"Error: No .h5 segment files in directory: {mouth_roi_speaker_path}")
                continue

            # Select a random segment file from this directory
            segment_file = random.choice(segment_files)
            print(f"Selected segment file: {segment_file}")

            # Construct paths for mouth_roi, audio, and video for this segment
            mouth_roi_file_path = Path(os.path.join(mouth_roi_speaker_path, segment_file))
            file_id = segment_file.split('.')[0]  # Extract the file ID, like '00039' from '00039.h5'
            audio_path = os.path.join(audio_speaker_path, f"{file_id}.wav")
            video_path = os.path.join(video_speaker_path, f"{file_id}.mp4")

            # Ensure paths exist
            if not os.path.exists(audio_path):
                print(f"Error: Audio file does not exist: {audio_path}")
                continue

            if not os.path.exists(video_path):
                print(f"Error: Video file does not exist: {video_path}")
                continue

            # For the first speaker ID, assign to audio1, for the second, assign to audio2
            if index == 0:
                audio1_path = audio_path
                mouth_roi1_path = mouth_roi_file_path
                video1_path = video_path
            else:
                audio2_path = audio_path
                mouth_roi2_path = mouth_roi_file_path
                video2_path = video_path

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

# Generate commands for each variation pair
def generate_commands(selections):
    commands = []
    for selection in selections:
        audio1_path = selection['audio1_path']
        audio2_path = selection['audio2_path']
        mouthroi1_path = selection['mouthroi1_path']
        mouthroi2_path = selection['mouthroi2_path']
        video1_path = selection['video1_path']
        video2_path = selection['video2_path']

        # Check if the files exist
        if not os.path.isfile(mouthroi1_path):
            print(f"Error: Mouth ROI file does not exist: {mouthroi1_path}")
            continue
        if not os.path.isfile(mouthroi2_path):
            print(f"Error: Mouth ROI file does not exist: {mouthroi2_path}")
            continue

        command = (
            f"python test_visualvoice.py --audio1_path {audio1_path} "
            f"--audio2_path {audio2_path} "
            f"--mouthroi1_path {mouthroi1_path} "
            f"--mouthroi2_path {mouthroi2_path} "
            f"--video1_path {video1_path} "
            f"--video2_path {video2_path} "
            f"--num_frames 64 --audio_length 2.55 --hop_size 160 --window_size 400 "
            f"--n_fft 512 --weights_lipreadingnet pretrained_models/lipreading_best.pth "
            f"--weights_facial pretrained_models/facial_best.pth "
            f"--weights_unet pretrained_models/unet_best.pth "
            f"--weights_vocal pretrained_models/vocal_best.pth "
            f"--lipreading_config_path configs/lrw_snv1x_tcn2x.json "
            f"--unet_output_nc 2 "
            f"--normalization "
            f"--mask_to_use pred "
            f"--visual_feature_type both "
            f"--identity_feature_dim 128 "
            f"--audioVisual_feature_dim 1152 "
            f"--visual_pool maxpool "
            f"--audio_pool maxpool "
            f"--compression_type none "
            f"--mask_clip_threshold 5 --hop_length 2.55 "
            f"--audio_normalization "
            f"--lipreading_extract_feature "
            f"--number_of_identity_frames 1 "
            f"--output_dir_root {output_dir_root}"
        )
        commands.append(command)
    return commands

# Main execution
if __name__ == "__main__":
    # Select random files and generate pairs
    selections = select_random_files_original(base_dir, n_pairs=1000)
    from pprint import pprint
    print(len(selections))

    # Generate commands for the selected variations
    commands = generate_commands(selections)
    for command in commands:
        try:
            subprocess.run(command, check=True, shell=True)
            print(f"Command executed successfully: {command}")
        except subprocess.CalledProcessError as e:
            # Handle errors in the command execution
            print(f"Error executing command: {command}\nError: {e}")
