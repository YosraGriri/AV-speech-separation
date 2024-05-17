import subprocess
import random
import os
# Base paths
audio_base_path = "../../data/VoxCeleb2/raw_audio"
audio_base_path = "../../data/simulated_RIR/VoxCeleb2/raw_audio"
audio_base_path = "E:/AV-speech-separation/data/VoxCeleb2/raw_audio_test"
video_base_path = "E:/AV-speech-separation/data/VoxCeleb2/mp4"
mouthroi_base_path = "E:/AV-speech-separation/data/VoxCeleb2/mouth_roi/unseen_unheard"
#mouthroi_base_path = "E:\AV-speech-separation\data\VoxCeleb2\mouth_roi\unseen_unheard\"


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

        # Check if the files exist
        if not os.path.isfile(f'{mouthroi_base_path}/{id1}/{session1}/{file1[:5]}.h5'):
            print(f"Error: Mouth ROI file does not exist: {mouthroi_base_path}/{id2}/{session2}/{file2[:5]}.h5")
            continue
        if not os.path.isfile(f'{mouthroi_base_path}/{id2}/{session2}/{file2[:5]}.h5'):
            print(f"Error: Mouth ROI file does not exist: {f'{mouthroi_base_path}/{id2}/{session2}/{file2[:5]}.h5'}")
            continue

        command = (f"python test.py --audio1_path {audio_base_path}/{id1}/{session1}/{file1}.wav "
                   f"--audio2_path {audio_base_path}/{id2}/{session2}/{file2}.wav "
                   f"--mouthroi1_path {mouthroi_base_path}/{id1}/{session1}/{file1[:5]}.h5 "
                   f"--mouthroi2_path {mouthroi_base_path}/{id2}/{session2}/{file2[:5]}.h5 "
                   f"--video1_path {video_base_path}/{id1}/{session1}/{file1[:5]}.mp4 "
                   f"--video2_path {video_base_path}/{id2}/{session2}/{file2[:5]}.mp4 "
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
                   f"--output_dir_root E:/AV-speech-separation/data/VoxCeleb2/results")
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

