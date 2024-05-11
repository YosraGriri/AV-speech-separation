from loading_data_utils import *
from utils.select_files_utils import select_random_files
from utils.beamformer import compute_psd_matrix, create_multichannel_psd
from beamformer.GEV_Beamformer import GEVBeamformer
from options.test_options import TestOptions  # Ensure this is correctly imported
from pprint import pprint
from beamform_it import aggregate_psd_matrices, GEVBeamformer

def main():
    opt = TestOptions().parse()
    beamformer = GEVBeamformer()
    psd_matrices_target = []
    psd_matrices_noise = []
    complex_spectrograms = []

    # Select files to process
    file_selections = select_random_files(opt.base_dir, opt.n_mic, voice_id=0)
    pprint(file_selections)

    for selection in file_selections:
        # Initialize the processor with specific file paths for this iteration
        processor = AudioProcessor(opt, beamformer,
                                   audio1_path=selection['audio1_path'],
                                   audio2_path=selection['audio2_path'],
                                   video1_path=selection['video1_path'],
                                   video2_path=selection['video2_path'],
                                   mouthroi1_path=selection['mouthroi1_path'],
                                   mouthroi2_path=selection['mouthroi2_path'])

        # Extract microphone identifiers from the audio file names
        mic_id_1 = os.path.basename(selection['audio1_path']).split('_')[1]  # Assumes format like '00386_mic0.wav'
        mic_id_2 = os.path.basename(selection['audio2_path']).split('_')[1]  # Assumes format like '00386_mic1.wav'

        # Processing loop
        samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
        sliding_window_start = 0
        print(processor.audio_length)
        while sliding_window_start + samples_per_window < processor.audio_length:
            sliding_window_end = sliding_window_start + samples_per_window
            mask1, mask2, audio_mix_spec, spec1, spec2 = processor.process_audio_segment(sliding_window_start, sliding_window_end)
            is_complex_mask_1 = np.iscomplexobj(mask1)
            is_complex_mask_2 = np.iscomplexobj(mask2)
            print(f"Is mask_prediction_1 complex? {is_complex_mask_1}")
            print(f"Is mask_prediction_2 complex? {is_complex_mask_2}")
            np.save(f'model_results/model_output/target_mask_{mic_id_1}.npy', mask1)
            np.save(f'model_results/model_output/noise_mask_{mic_id_2}.npy', mask2)
            np.save(f'model_results/model_output/mixture_spec_{mic_id_1}.npy', audio_mix_spec)
            sliding_window_start += int(opt.hop_length * opt.audio_sampling_rate)
            # Compute PSD matrices and handle results
            psd_target = compute_psd_matrix(audio_mix_spec, mask1)
            psd_noise = compute_psd_matrix(audio_mix_spec, mask2)
        psd_matrices_target.append(psd_target)
        print(len(psd_matrices_target))
        psd_matrices_noise.append(psd_noise)
        complex_spectrograms.append(spec1)

        # Handle final results
        energy_diagonal_target = np.abs(np.diagonal(psd_target))**2
        energy_diagonal_noise = np.abs(np.diagonal(psd_noise))**2
        output_dir = processor.save_results(
            selection['audio1_path'], selection['audio2_path'],
            selection['video1_path'],selection['video2_path'])

        multichannel_psd_target = aggregate_psd_matrices(psd_matrices_target, complex_spectrograms)
        np.save('psd_matrices/psd_matrix_target.npy', multichannel_psd_target)
        multichannel_psd_noise = aggregate_psd_matrices(psd_matrices_noise, complex_spectrograms)
        np.save('psd_matrices/psd_matrix_noise.npy', multichannel_psd_noise)
        #Diagonals
        diagonal_elements_target = np.array(
            [np.diagonal(multichannel_psd_target[i]) for i in range(multichannel_psd_target.shape[0])])
        energy_diagonal_target_multichannel = np.abs(diagonal_elements_target) ** 2
        diagonal_elements_noise = np.array(
            [np.diagonal(multichannel_psd_noise[i]) for i in range(multichannel_psd_noise.shape[0])])
        energy_diagonal_noise_multichannel = np.abs(diagonal_elements_noise) ** 2



        # Define the output file paths with mic identifiers
        output_psd_target_path = os.path.join(output_dir, f'squared_magnitudes_target_{mic_id_1}.npy')
        output_psd_noise_path = os.path.join(output_dir, f'squared_magnitudes_noise_{mic_id_2}.npy')
        output_psd_target_multichannel = os.path.join(output_dir, f'squared_magnitudes_target_multichannel.npy')
        output_psd_noise_multichannel = os.path.join(output_dir, f'squared_magnitudes_noise_multichannel.npy')
        np.save(output_psd_noise_path, energy_diagonal_noise)
        np.save(output_psd_target_path, energy_diagonal_target)
        np.save(output_psd_target_multichannel, energy_diagonal_target_multichannel)
        np.save(output_psd_noise_multichannel, energy_diagonal_noise_multichannel)

        #Beamform it baby!
        #gev_beamformer = GEVBeamformer()
        #bf_vector = gev_beamformer.get_beamforming_vector(multichannel_psd_target, multichannel_psd_noise)


if __name__ == '__main__':
    main()
