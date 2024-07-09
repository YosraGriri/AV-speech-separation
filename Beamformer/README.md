# GEV Beamforming Project

This project implements Generalized Eigenvalue (GEV) beamforming

## Directory Structure

- `GEV_Beamformer.py`: Main script for processing audio files using GEV beamforming on the magnitude of the predicted Masks from VisualVoice.
- `GEV_Beamformer_ideal.py`: Script for ideal beamforming scenarios.
- `Utils/beamform_it.py`: Contains functions for beamforming operations.
- `Utils/signal_processing.py`: Utility functions for signal processing.
- `Utils/utils.py`: Helper functions for loading and evaluating audio files.

## Usage

### GEV_Beamformer.py

This script processes audio files using GEV beamforming.

1. Set the appropriate paths and configurations in the script.
2. Run the script:
    ```bash
    python GEV_Beamformer.py
    ```
This will process the audio files in the specified directory and save the results to an Excel file.
### GEV_Beamformer_librosa.py

This script processes audio files using an invertible Matrix A:
    ```python
    matrix_A = np.array([[1, 1], [1, -1]])
    ```

1. Set the target, noise, and noisy speech file paths in the script.
2. Run the script:
    ```bash
    python GEV_Beamformer_ideal.py
    ```
This will process the audio files in the specified directory and save the results to an Excel file.
### Functions in beamform_it.py

- `blind_analytic_normalization(vector, noise_psd_matrix, eps=1e-16)`: Reduces distortions in the beamformed output.
- `get_power_spectral_density_matrix(observation, mask=None, normalize=True)`: Calculates the weighted power spectral density matrix.

### Functions in signal_processing.py

- `segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0)`: Segments an array along a specified axis.
- `stft(time_signal, time_dim=None, size=512, shift=256, window=signal.blackman)`: Performs Short-Time Fourier Transform on a time-domain signal.

### Functions in utils.py

- `load_wav_files(base_path)`: Loads WAV files from a specified directory, categorizing them into targets, noise, and mixtures based on filename patterns.


## Acknowledgment

This Beamformer is implemented following the method by Heymann et al.  
Heymann, J., Drude, L., & Haeb-Umbach, R. (2016, March). Neural network based spectral mask estimation for acoustic beamforming. In 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 196-200). IEEE.


