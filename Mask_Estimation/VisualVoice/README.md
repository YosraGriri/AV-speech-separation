# Mask Estimation using VisualVoice

This repository contains scripts for using the pre-trained VisualVoice model to predict complex ideal ratio masks.

## Prerequisites

Before using the scripts, ensure you have completed the following steps:

1. Download the pre-trained models from the [VisualVoice GitHub repository](https://github.com/facebookresearch/VisualVoice) and place them in the `pretrained_models` folder.
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

The separated audio files and masks will be saved in the `../../data/test/results/original/complex/` directory for now or any other directory of your choice. 

## Acknowledgements
Some of the code is borrowed or adapted from the VisualVoice repository: [VisualVoice GitHub repository](https://github.com/facebookresearch/VisualVoice).






