# Room Impulse Response (RIR) Generation using Pyroom Acoustics

## Description

This project generates room impulse responses (RIR) using the Pyroom Acoustics library. The main script processes audio sources from specified directories and simulates their acoustics in virtual room environments with defined characteristics.

## Installation

To install the necessary dependencies, run:
```bash
pip install -r ../requirements.txt

## Usage
python main.py --output_path <output_path> --source_dir <source_dir> [other arguments]
for example, to run this on the samples published in the ../data/test folder:
python Scripts/main.py --source_dir  "../../data/test/raw_audio"    --output_path  "../../data/test/simulated/"

## File Descriptions of the files in Scripts

- **main.py**:  Parses arguments and initiates the room simulation process.
- **generate_sample_all.py**: Contains the function to generate room simulations for all folders with voice samples.
- **get_voice.py**: Contains functions to extract voice data and associated information from the source directories.
- **plot.py**: Contains a function to visualize the room setup, including sources and microphones.
- **simulate_room.py**: Contains functions to create and configure the room environment for simulations.

## Acknowledgments

The RIR Generation is implemented using the python library Pyroomacoustics.
> R. Scheibler, E. Bezzam, I. Dokmanić, *Pyroomacoustics: A Python package for audio room simulations and array processing algorithms*, Proc. IEEE ICASSP, Calgary, CA, 2018.

  


