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

## Usage

To separate audio using VisualVoice, you can run the following scripts:

1. Using simulated RIR:

    ```bash
    python generate_test_commands_simulated.py
    ```

2. Without simulated RIR:

    ```bash
    python generate_test_commands.py
    ```
These Scripts will save a separated audio for each microphone along with the prediced cIRM.

## Acknowledgement

The files in the `configs/models/utils/data/options` directories were downloaded from the VisualVoice GitHub repository. Some files have been slightly modified to suit the thesis requirements.

The mask estimation in this thesis uses the pre-trained VisualVoice model. 
