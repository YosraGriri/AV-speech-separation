# Audio Separation Evaluation

## Overview

This repository contains scripts for evaluating audio separation quality using various metrics including SDR, SIR, SAR, PESQ, and STOI. The scripts process directories of audio files, compute the metrics, and save the results to an Excel file.

# Usage

## To evaluate the Separation of the original VoxCeleb2 separation 
### eval_VisVoi_original.py


This script evaluates the separation quality of audio files in a specified directory.

1. Set the `test_dir` variable in the `main()` function to the path of the directory containing the audio files.
2. Run the script:

```bash
python eval_VisVoi_original.py
``` 
## To evaluate the Separation of the simulated VoxCeleb2:
### eval_VisVoi_original.py

This script evaluates the separation quality of audio files in a specified directory.

1. Set the `test_dir` variable in the `main()` function to the path of the directory containing the audio files.
2. Run the script:

```bash
python eval_VisVoi_simulated.py
``` 