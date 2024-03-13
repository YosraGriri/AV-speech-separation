from pathlib import Path


# Function to extract filenames from paths and generate a unique combination ID
def generate_voice_combination_id(video_paths):
    # Extract the filenames from the paths
    print(len(video_paths))
    filenames = [Path(path).name for path in video_paths]
    # Sort and join the filenames to create a unique identifier
    voice_combination_id = "_".join(sorted(filenames))
    return voice_combination_id


def generate_voice_combination_id_vox(paths):
    # Extract IDs from the part of the path before the last and also prepare the last three parts
    ids = [Path(path).parts[-2] for path in paths]  # Extract IDs
    last_three_parts = ["/".join(Path(path).parts[-3:]) for path in paths]  # Get the last three parts
    # Print the number of paths processed
    print(len(paths))
    # Sort and join the IDs to create a unique identifier
    voice_combination_id = "_".join(sorted(ids))
    # Return both the unique identifier and the last three parts of each path
    return voice_combination_id, last_three_parts


def get_voice_combination_id(video_paths):
    # Extract filenames from each path
    filenames = [Path(path).name for path, _ in video_paths]
    # Sort filenames to ensure consistent ordering
    sorted_filenames = sorted(filenames)
    # Join sorted filenames to create a unique identifier
    voice_combination_id = "_".join(sorted_filenames)
    return voice_combination_id
