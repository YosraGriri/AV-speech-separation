import os
import random

# Base paths
mouthroi_base_path = "../../data/VoxCeleb2/mouth_roi/unseen_unheard_test"

# Collect all available IDs, sessions, and filenames based on mouth ROI files
available_files = []
for id_dir in os.listdir(mouthroi_base_path):
    for session_dir in os.listdir(os.path.join(mouthroi_base_path, id_dir)):
        for file in os.listdir(os.path.join(mouthroi_base_path, id_dir, session_dir)):
            file_name = file.split('.')[0]  # Assuming the file format is filename.extension
            available_files.append(f"{id_dir}/{session_dir}/{file_name}")
            print(available_files)

# Shuffle the list to randomize the order of files
random.shuffle(available_files)

# Save pairs to a file
with open("valid_pairs.txt", "w") as file:
    for file_name in available_files:
        file.write(f"{file_name}\n")
