# Master's Thesis Project: Integrating Neural Networks with Beamforming for AVSS

This README file provides an overview of the Audio-Visual Speech Source Separation (AVSS) project for my master's thesis. It includes a brief description of the thesis, its structure, and instructions for usage and setup.

## Table of Contents

- [Description](#description)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [TBA](#tba)

## Description

This project focuses on the development of an audio-visual mask-based beamformer.

## Project Structure

The project is organized into the following directories and files:

- `data/`: Contains a sample from the VoxCeleb2 dataset for testing.
- `Simulation/`: Simulates multi-channel data using Pyroomacoustics.
- `mask_estimation/`: Includes the scripts for mask estimation.
- `Beamformer/`: Contains the implementation of the GEV Beamformer.
- `Evaluation/`: Source code for the audio-visual speech source separation system.

Each of these folders contains a README file with further explanations on how they can be used. The directories should be used in the order listed: first Multi-channel simulation, then mask estimation, and finally the Beamformer.

## Requirements

1. Clone this repository to your local machine:

   ```bash
   git clone git@github.com:YosraGriri/AV-speech-separation.git

2. Install all necessary libraries:
pip install -r requirements.txt
