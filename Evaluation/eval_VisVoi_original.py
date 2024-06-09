import os
import librosa
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from pesq import pesq
from pystoi import stoi
import mir_eval
from scipy.io import wavfile

def getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt):
    duration = int(2.55 *16000)
    print(duration)
    print(len(audio1))
    print(len(audio1_gt))
    reference_sources = np.concatenate((np.expand_dims(audio1_gt[:duration], axis=0), np.expand_dims(audio2_gt[:duration], axis=0)), axis=0)
    estimated_sources = np.concatenate((np.expand_dims(audio1[:duration], axis=0), np.expand_dims(audio2[:duration], axis=0)), axis=0)
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, False)
    return np.mean(sdr), np.mean(sir), np.mean(sar)

def process_folder(results_dir, audio_sampling_rate):
    # Find the wav files in the directory
    wav_files = [f for f in os.listdir(results_dir) if f.endswith('.wav')]

    print(f'The results_dir is: {results_dir}')

    # Filter out mixed files
    separated_files = [f for f in wav_files if 'separated' in f]
    ground_truth_files = [f for f in wav_files if 'separated' not in f and 'mixed' not in f]

    if len(separated_files) != 2 or len(ground_truth_files) != 2:
        print(f"Skipping folder {results_dir} due to mismatched file counts.")
        return None

    # Sort files to match separated with ground truth
    separated_files.sort()
    ground_truth_files.sort()

    # Assume naming convention: 001.wav -> 001_separated.wav
    audio1_gt_path = Path(os.path.join(results_dir, ground_truth_files[0]))
    #print()
    audio2_gt_path = Path(os.path.join(results_dir, ground_truth_files[1]))
    audio1_path = Path(os.path.join(results_dir, separated_files[0]))
    audio2_path = Path(os.path.join(results_dir, separated_files[1]))

    print(f'The audio1_gt_path is: {audio1_gt_path}')
    print(f'The audio2_gt_path is: {audio2_gt_path}')
    print(f'The audio1_path is: {audio1_path}')
    print(f'The audio2_path is: {audio2_path}')

    try:
        _, audio1 = wavfile.read(audio1_path)
        _, audio2 = wavfile.read(audio2_path)
        _, audio1_gt = wavfile.read(audio1_gt_path)
        _, audio2_gt = wavfile.read(audio2_gt_path)

        sdr, sir, sar = getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt)
        print(f'SDR: {sdr}\n'
              f'SIR: {sir}\n'
              f'SAR: {sar}')

        stoi_score1 = stoi(audio1_gt, audio1, audio_sampling_rate, extended=False)
        stoi_score2 = stoi(audio2_gt, audio2, audio_sampling_rate, extended=False)
        stoi_score = (stoi_score1 + stoi_score2) / 2
        print(f'STOI score: {stoi_score}')

        pesq_score1 = pesq(audio_sampling_rate, audio1, audio1_gt, 'wb')
        pesq_score2 = pesq(audio_sampling_rate, audio2, audio2_gt, 'wb')
        pesq_score = (pesq_score1 + pesq_score2) / 2
        print(f'PESQ score is: {pesq_score}')

        return sdr, sir, sar, pesq_score, stoi_score
    except Exception as e:
        print(f"Error processing folder {results_dir}: {e}")
        return None

def main():
    test_dir = "E:/AV-speech-separation/data/VoxCeleb2/results/original/no_phase/"
    audio_sampling_rate = 16000
    mic_index = 0

    # DataFrame to store all results
    results_df = pd.DataFrame(columns=['Folder', 'SDR', 'SIR', 'SAR', 'PESQ', 'STOI'])
    test_dirs = os.listdir(test_dir)
    i=1951
    j=2000
    

    # Loop over folders in the test directory
    for folder_name in test_dirs[i:j]:
        folder_path = os.path.join(test_dir, folder_name)
        print(folder_path)
        if os.path.isdir(folder_path):
            print('Processing folder:', folder_name)
            result = process_folder(folder_path, audio_sampling_rate)
            if result is None:
                continue
            sdr, sir, sar, pesq_score, stoi_score = result
            results_df = pd.concat([results_df, pd.DataFrame([{
                'Folder': folder_name,
                'SDR': sdr,
                'SIR': sir,
                'SAR': sar,
                'PESQ': pesq_score,
                'STOI': stoi_score
            }])], ignore_index=True)

    # Calculate averages and add them to the DataFrame
    averages = results_df.mean(numeric_only=True)
    averages['Folder'] = 'Average'
    averages_df = pd.DataFrame([averages])
    results_df = pd.concat([results_df, averages_df], ignore_index=True)
    print(test_dir)

    # Save to Excel
    excel_path = os.path.join(test_dir, f'/evaluation_results_no_phase_{i}_{j}.xlsx')
    print(excel_path)
    results_df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")

if __name__ == '__main__':
    main()


