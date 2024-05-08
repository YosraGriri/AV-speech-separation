import subprocess
import random

# Base paths
audio_base_path = "../../data/VoxCeleb2/raw_audio"
audio_base_path = "../../data/simulated_RIR/VoxCeleb2/raw_audio"
video_base_path = "../../data/VoxCeleb2/mp4"
mouthroi_base_path = "../../data/VoxCeleb2/mouth_roi/unseen_unheard_test"


# Function to read pairs from file and generate variations
def read_pairs_and_generate_variations(file_path, n_pairs=20, n_mics=4):
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file.readlines()]

    # Sample n_pairs*2 files to form pairs
    if len(lines) < n_pairs * 2:
        print("Not enough entries for the requested number of pairs.")
        return []

    selected_lines = random.sample(lines, n_pairs * 2)  # Ensuring unique pairs by sampling without replacement
    pairs = [(selected_lines[i], selected_lines[i + 1]) for i in range(0, len(selected_lines), 2)]

    # Generate variations for each pair, ensuring the same mic is used for both IDs in a pair
    variations = []
    for (id1, id2) in pairs:
        for mic in range(n_mics):
            variations.append((f"{id1}_mic{mic}_voice1", f"{id2}_mic{mic}_voice1"))
    return variations


# Generate commands for each variation pair
def generate_commands(variations):
    commands = []
    for id1_variation, id2_variation in variations:

        id1, session1, file1 = id1_variation.split('/')
        id2, session2, file2 = id2_variation.split('/')
        print (f'{audio_base_path}/{id1}/{session1}/{file1[:5]}.wav')

        command = f"python test.py --audio1_path {audio_base_path}/{id1}/{session1}/{file1}.wav --audio2_path {audio_base_path}/{id2}/{session2}/{file2}.wav --mouthroi1_path {mouthroi_base_path}/{id1}/{session1}/{file1[:5]}.h5 --mouthroi2_path {mouthroi_base_path}/{id2}/{session2}/{file2[:5]}.h5 --video1_path {video_base_path}/{id1}/{session1}/{file1[:5]}.mp4 --video2_path {video_base_path}/{id2}/{session2}/{file2[:5]}.mp4 --num_frames 64 --audio_length 2.55 --hop_size 160 --window_size 400 --n_fft 512 --weights_lipreadingnet pretrained_models/lipreading_best.pth --weights_facial pretrained_models/facial_best.pth --weights_unet pretrained_models/unet_best.pth --weights_vocal pretrained_models/vocal_best.pth --lipreading_config_path configs/lrw_snv1x_tcn2x.json --unet_output_nc 2 --normalization --mask_to_use pred --visual_feature_type both --identity_feature_dim 128 --audioVisual_feature_dim 1152 --visual_pool maxpool --audio_pool maxpool --compression_type none --mask_clip_threshold 5 --hop_length 2.55 --audio_normalization --lipreading_extract_feature --number_of_identity_frames 1 --output_dir_root test/simulated_RIR/"
        commands.append(command)
    return commands


# Main execution
if __name__ == "__main__":
    # Read pairs from the file and generate microphone variations
    variations = read_pairs_and_generate_variations("valid_pairs.txt", 100)

    # Generate commands for the selected variations
    commands = generate_commands(variations)
    for command in commands[:5]:
        try:
            subprocess.run(command, check=True, shell=True)
            print(f"Command executed successfully: {command}")
        except subprocess.CalledProcessError as e:
            # Handle errors in the command execution
            print(f"Error executing command: {command}\nError: {e}")

