# Mask Estimation using VisualVoice

This repository contains scripts for using the pre-trained VisualVoice model to predict complex ideal ratio masks.

## Prerequisites

Before using the scripts, ensure you have completed the following steps:

1. Download the pre-trained models from the [VisualVoice GitHub repository](https://github.com/facebookresearch/VisualVoice) and place them in the `pre-trained model` folder.
2. Install the required libraries from the main folder of this repository:

    ```bash
    cd ../../
    pip install -r requirements.txt
    ```

## Dataset

Before running the code, download the VoxCeleb2 dataset. You can request the download link by using your university email to contact [here](https://mm.kaist.ac.kr/datasets/voxceleb).

The pre-processed mouth ROIs can be downloaded from the [VisualVoice GitHub repository](https://github.com/facebookresearch/VisualVoice).

## Usage

To generate separated audio files using VisualVoice, you can run the following scripts:

1. Using simulated RIR:

    ```bash
    python generate_test_commands_simulated.py
    ```

2. Without simulated RIR:

    ```bash
    python generate_test_commands.py
    ```

## Data Paths

Before running the scripts, ensure that the data paths are set correctly according to your use:

