# Master's Thesis Project: Integrating Neural Networks with Beamforming for AVSS

This README file provides an overview of the Audio-Visual Speech Source Separation (AVSS) project for my master's thesis. It includes a brief description of the thesis, its structure, and instructions for usage and setup.

## Table of Contents

- [Description](#description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#Usage)


## Description
Abstract: 
In a noisy environment, the perceived speech is technically a mixture of sounds
coming from different sources. Yet the human ear manages to filter out unwanted
sounds in most scenario. Speech separation seeks to achieve this human abili
tiy computationally. In fact, during the last deacades, speech separation have
seen extensive research. And many methods have been developed from traditional
beamforming to audio-visual neural networks. This thesis aims to solve the task of
speech separation via intergrating an audio-visual neural networks with beamform-
ing. Starting with simulating multichannel data using the image source method.
Then, a mask-based audio-visual beamformer is implemented. Namely, the state
of the art VisualVoice model is chosen to estimate complex ideal ratio masks on
each channel separately. The different predicted masks are then combined, and
their median is calculated to form a single mask. A generalized eigenvector beam-
former is implemented and applied on the magnitude of this mask, along with the
complex spectrogram. The reproduced SDR, SIR, and SAR values for VisualVoice
on VoxCeleb2 are almost the same as those presented in the original paper. The
model succeeds in separating speech from the simulated data, yet a drop in the
metrics is noticed due to the realistic settings. The beamformer shows good results
in perfect settings but poor results using the predicted mask, which is due to the
importance of the phase information that had to be disregarded when using this
beamformer.



## Project Structure

The project is organized into the following directories and files:

- `data/`: Contains a sample from the VoxCeleb2 dataset for testing.
- `Simulation/`: Simulates multi-channel data using Pyroomacoustics.
- `mask_estimation/`: Includes the scripts for mask estimation.
- `Beamformer/`: Contains the implementation of the GEV Beamformer.
- `Evaluation/`: Source code for the audio-visual speech source separation system.

The folders: Simulation, mask_estimation/visualvoice/, Beamformer and Evaluation contain each a README file with further explanations on how they can be used. 

## installation

1. Clone this repository to your local machine:

   ```bash
   git clone git@github.com:YosraGriri/AV-speech-separation.git

2. Install all necessary libraries:
```bash
pip install -r requirements.txt
```

## Usage
The directories should be used in the order listed: first Multi-channel simulation, then mask estimation, and finally the Beamformer.

