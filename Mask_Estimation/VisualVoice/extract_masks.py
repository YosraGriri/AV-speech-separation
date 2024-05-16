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
        mic_id_1 = os.path.basename(selection['audio1_path']).split('_')[1]

        # Processing loop
        output_dir = processor.save_results(selection['audio1_path'], selection['audio2_path'],
                                          selection['video1_path'], selection['video2_path'])
        mask_dir = os.path.join(output_dir, 'masks')
        os.makedirs(mask_dir, exist_ok=True)
        print(f"Is mic_id_1 complex? {mic_id_1}")


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
            np.save(os.path.join(mask_dir, f'target_mask_{mic_id_1}.npy'), mask1)
            np.save(os.path.join(mask_dir, f'noise_mask_{mic_id_1}.npy'), mask2)
            np.save(os.path.join(mask_dir, f'mixture_spec_{mic_id_1}.npy'), audio_mix_spec)
            sliding_window_start += int(opt.hop_length * opt.audio_sampling_rate)

            output_dir = processor.save_results(selection['audio1_path'],
                                                selection['audio2_path'],
                                                selection['video1_path'],
                                                selection['video2_path'])

if __name__ == '__main__':
    main()
