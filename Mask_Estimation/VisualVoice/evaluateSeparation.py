#!/usr/bin/env python

import os
import librosa
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import mir_eval.separation
from pypesq import pesq
from pystoi import stoi

def getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt):
    reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0)), axis=0)
    estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, False)
    return np.mean(sdr), np.mean(sir), np.mean(sar)

def process_folder(results_dir, audio_sampling_rate):
    audio_path1 = Path(os.path.join(results_dir, 'audio1_separated.wav'))
    audio_path2 = Path(os.path.join(results_dir, 'audio2_separated.wav'))
    audio1, _ = librosa.load(audio_path1, sr=audio_sampling_rate)
    audio2, _ = librosa.load(audio_path2, sr=audio_sampling_rate)
    audio1_gt, _ = librosa.load(os.path.join(results_dir, 'audio1.wav'), sr=audio_sampling_rate)
    audio2_gt, _ = librosa.load(os.path.join(results_dir, 'audio2.wav'), sr=audio_sampling_rate)

    sdr, sir, sar = getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt)
    print (f'sdr: {sdr}\n'
           f'sir: {sir}\n'
           f'sar: {sar}')
    stoi_score1 = stoi(audio1_gt, audio1, audio_sampling_rate, extended=False)
    stoi_score2 = stoi(audio2_gt, audio2, audio_sampling_rate, extended=False)
    stoi_score = (stoi_score1 + stoi_score2) / 2
    print(f'stoi_score: {stoi_score}')
    try:
        pesq_score1 = pesq(audio1, audio1_gt, audio_sampling_rate)
        pesq_score2 = pesq(audio2, audio2_gt, audio_sampling_rate)
        pesq_score = (pesq_score1 + pesq_score2) / 2
    except Exception as e:
        print(f"Error during PESQ calculation: {e}")
        pesq_score = 0
    print(f'pesq score is: {pesq_score}')
    return sdr, sir, sar, pesq_score, stoi_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--audio_sampling_rate', type=int, default=16000)
    args = parser.parse_args()

    # DataFrame to store all results
    results_df = pd.DataFrame(columns=['Folder', 'SDR', 'SIR', 'SAR', 'PESQ', 'STOI'])

    # Loop over folders in the test directory
    for folder_name in os.listdir(args.test_dir):
        #print(len(os.listdir(args.test_dir)))
        folder_path = os.path.join(args.test_dir, folder_name)
        if os.path.isdir(folder_path):
            print('HELLLO')
            sdr, sir, sar, pesq_score, stoi_score = process_folder(folder_path, args.audio_sampling_rate)
            print ('youpii')
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

    # Save to Excel
    excel_path = os.path.join(args.test_dir, 'evaluation_results.xlsx')
    results_df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")

if __name__ == '__main__':
    main()