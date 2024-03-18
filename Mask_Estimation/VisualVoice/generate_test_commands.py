import subprocess
import random
# Base paths
audio_base_path = "../../data/VoxCeleb2/raw_audio"
video_base_path = "../../data/VoxCeleb2/mp4"
mouthroi_base_path = "../../data/VoxCeleb2/mouth_roi/unseen_unheard_test"


# Function to read pairs from file
def read_pairs_from_file(file_path, n_pairs=20):
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file.readlines()]

    # Sample n_pairs*2 files to form pairs
    if len(lines) < n_pairs * 2:
        print("Not enough entries for the requested number of pairs.")
        return []

    selected_lines = random.sample(lines, n_pairs * 2)  # Ensuring unique pairs by sampling without replacement
    pairs = [(selected_lines[i], selected_lines[i + 1]) for i in range(0, len(selected_lines), 2)]
    return pairs


# Generate commands for each pair
def generate_commands(pairs):
    commands = []
    for pair in pairs:
        id1, session1, file1 = pair[0].split('/')
        id2, session2, file2 = pair[1].split('/')
        command = f"python test.py --audio1_path {audio_base_path}/{id1}/{session1}/{file1}.wav --audio2_path {audio_base_path}/{id2}/{session2}/{file2}.wav --mouthroi1_path {mouthroi_base_path}/{id1}/{session1}/{file1}.h5 --mouthroi2_path {mouthroi_base_path}/{id2}/{session2}/{file2}.h5 --video1_path {video_base_path}/{id1}/{session1}/{file1}.mp4 --video2_path {video_base_path}/{id2}/{session2}/{file2}.mp4 --num_frames 64 --audio_length 2.55 --hop_size 160 --window_size 400 --n_fft 512 --weights_lipreadingnet pretrained_models/lipreading_best.pth --weights_facial pretrained_models/facial_best.pth --weights_unet pretrained_models/unet_best.pth --weights_vocal pretrained_models/vocal_best.pth --lipreading_config_path configs/lrw_snv1x_tcn2x.json --unet_output_nc 2 --normalization --mask_to_use pred --visual_feature_type both --identity_feature_dim 128 --audioVisual_feature_dim 1152 --visual_pool maxpool --audio_pool maxpool --compression_type none --mask_clip_threshold 5 --hop_length 2.55 --audio_normalization --lipreading_extract_feature --number_of_identity_frames 1 --output_dir_root test"
        commands.append(command)
    return commands


# Read pairs from the file
pairs = read_pairs_from_file("valid_pairs.txt", 100)

# Generate commands for the selected pairs
commands = generate_commands(pairs)
from pprint import pprint
# Example: print or execute commands
for command in commands:
    try:
        # Execute command
        subprocess.run(command, check=True, shell=True)
        print(f"Command executed successfully: {command}")
    except subprocess.CalledProcessError as e:
        # Handle errors in the command execution
        print(f"Error executing command: {command}\nError: {e}")

