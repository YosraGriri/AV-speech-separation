import sys
import pandas as pd
sys.path.append('../utils')
from utils.signal_processing import*
from utils.beamform_it import *
from utils.utils import *
import soundfile as sf


results_path = "../data/VoxCeleb2/raw_audio_test/result/"
results_path = "E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
results_path = "../data/test/results/simulated/complex"
root_dir = results_path  # Root directory containing all combinations
duration = int(2.55 * 16000)
output_excel = "beamforming_results.xlsx"
#output_excel = os.path.join(results_path, beamforming_results.xlsx)
# Prepare a list to store results
results = []
i = 0
j = 1

# Paths and configurations
results_path = "../data/test/results/simulated/complex"
#results_path = "E:/AV-speech-separation/data/VoxCeleb2/results/simulated/"
root_dir = results_path  # Root directory containing all combinations
duration = int(2.55 * 16000)
n_mic = 4
output_excel = f"beamforming_simulated_results_NrOfMic_{n_mic}_{i}_{j}.xlsx"
output_excel = os.path.join(results_path, f"beamforming_simulated_results_NrOfMic_{n_mic}_{i}_{j}.xlsx")

# Loop over each combination directory
for combination_id in os.listdir(root_dir)[i:j]: 
    base_path = os.path.join(root_dir, combination_id)
    print(f"Processing {base_path}")
    
    if not os.path.isdir(base_path):
        print(f"Skipping {base_path}, not a directory.")
        continue
    
    try:
        # Load WAV and numpy files
        targets, noises, mixtures = load_wav_files(base_path)
        targets = targets [:n_mic]
        noises = noises [:n_mic]
        mixtures = mixtures [:n_mic]
        target_masks, noise_masks, _ = load_numpy_files(base_path)
        target_masks = target_masks[:n_mic]
        noise_masks = noise_masks[:n_mic]
      
        # Check if there are at least 4 targets
        if len(targets) < n_mic:
            print(f"Skipping {combination_id} due to insufficient targets")
            continue

        # Process masks
        X_mask, N_mask = process_and_stack_masks(noise_masks, target_masks)
        
        print(f"Shape of X_mask: {X_mask.shape}, Shape of N_mask: {N_mask.shape}")

        
        # Mix and STFT audio files
        mixed_signals, multi_channel_clean, multi_channel_noise, Y, stft_clean, stft_noise = mix_and_stft_audio_files(mixtures, targets, noises)
        print(f"Shape of mixed_signals: {mixed_signals.shape}")
       
        mic_results = []
        max_sdr, max_sir, max_sar, max_pesq, max_stoi = -np.inf, -np.inf, -np.inf, -np.inf, -np.inf
        for mic in range(mixed_signals.shape[0]):  # Loop over each microphone channel
            print(f"Processing microphone {mic} in {combination_id}")
        
            # Apply GEV beamforming
            Y_hat = gev_wrapper_on_masks(Y, N_mask, X_mask)
            Y_noise_hat = gev_wrapper_on_masks(Y, X_mask, N_mask)
            X_hat = librosa.istft(Y_hat, hop_length=160, win_length=400)
            X_noise = librosa.istft(Y_noise_hat, hop_length=160, win_length=400)
            
            print(f"Shape of Y_hat: {Y_hat.shape}, Shape of Y_noise_hat: {Y_noise_hat.shape}")

            stft_clean = stft_clean.transpose((1, 0, 2))
            stft_noise = stft_noise.transpose((1, 0, 2))

            # Calculate metrics
            sdr, sir, sar = getSeparationMetrics(X_hat, X_noise, targets[mic][:duration], noises[mic][:duration])
            print(f"Mic {mic} - SDR: {sdr}, SIR: {sir}, SAR: {sar}")
            stoi_score1 = stoi(targets[mic][:duration], X_hat, 16000, extended=False)
            stoi_score2 = stoi(noises[mic][:duration], X_noise, 16000, extended=False)
            stoi_score = (stoi_score1 + stoi_score2) / 2
            print(f'STOI score: {stoi_score}')

            pesq_score1 = pesq(16000, targets[mic][:duration], X_hat, 'wb')
            pesq_score2 = pesq(16000, noises[mic][:duration], X_noise, 'wb')
            pesq_score = (pesq_score1 + pesq_score2) / 2
            print(f'PESQ score is: {pesq_score}')
            
            mic_results.extend([sdr, sir, sar, pesq_score, stoi_score])

            # Update max values
            if sdr > max_sdr:
                max_sdr = sdr
            if sir > max_sir:
                max_sir = sir
            if sar > max_sar:
                max_sar = sar
            if pesq_score > max_pesq:
                max_pesq = pesq_score
            if stoi_score > max_stoi:
                max_stoi = stoi_score
        
        # Append results for all microphones and max values
        results.append([combination_id] + mic_results + [max_sdr, max_sir, max_sar, max_pesq, max_stoi])
        print(f"Results for {combination_id} appended successfully.")
        target_id, noise_id = extract_ids(combination_id)  
        
        sf.write(f"../data/test/results/simulated/GEV/GEV_target_{target_id}.wav", X_hat[:duration], 16000)
        sf.write(f"../data/test/results/simulated/GEV/GEV_noise_{noise_id}.wav", X_noise[:duration], 16000)
        
    except Exception as e:
        print(f"Error processing {combination_id}: {e}")
        continue


# Check if results list is not empty before creating the DataFrame
if results:
    # Prepare column names for the DataFrame
    columns = ['ID']
    for mic in range(mixed_signals.shape[0]):
        columns.extend([f'SDR_{mic}', f'SIR_{mic}', f'SAR_{mic}', f'PESQ_{mic}', f'STOI_{mic}'])
    columns.extend(['Max_SDR', 'Max_SIR', 'Max_SAR', 'Max_PESQ', 'Max_Stoi'])

    # Save results to Excel
    df_results = pd.DataFrame(results, columns=columns)
    df_results.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")
else:
    print("No results to save.")
