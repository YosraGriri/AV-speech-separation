from test_utils import*
import matplotlib.pyplot as plt
from pprint import pprint
from beamformer.GEV_Beamformer import GEVBeamformer
import subprocess


def compute_psd_matrix(observation, mask=None, normalize=True):
    """
    Calculate the weighted power spectral density matrix using the observation and mask.
    """
    print(f'The observation shape is: {observation[0, 0].shape}')
    print(f'The mask shape is: {mask[0, 0].shape}')
    bins, sensors, frames = observation.shape
    observation = observation[:, :-1, :]

    if mask is None:
        mask = np.ones((bins, frames))
    if mask.ndim == 2:
        mask = mask[:, np.newaxis, :]
    else:
        # Ensure the mask is compatible with the observation's dimensions.
        mask = mask[:bins, :, :frames]

    psd = np.einsum('...dt,...et->...de', mask * observation, observation.conj())

    #psd = np.einsum('...dt,...et->...de', mask[0, 0] * observation[0,0], mask[0, 1] * observation[0, 1] .conj())
    if normalize:
        normalization = np.sum(mask, axis=-1, keepdims=True)
        psd /= normalization
    return psd
def main():
    mask_target = []
    mask_noise = []
    opt = TestOptions().parse()
    beamformer = GEVBeamformer()
    processor = AudioProcessor(opt, beamformer)

    samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
    sliding_window_start = 0
    print(processor.audio_length)
    loop_counter = 0
    while sliding_window_start + samples_per_window < processor.audio_length:
        sliding_window_end = sliding_window_start + samples_per_window
        mask1, mask2, audio_mix_spec = processor.process_audio_segment(sliding_window_start, sliding_window_end)
        sliding_window_start += int(opt.hop_length * opt.audio_sampling_rate)
        mask_target.append(mask1)
        mask_noise.append(mask2)
        loop_counter += 1
        print(f'The observation shape is: {audio_mix_spec.shape}')
        psd_target = compute_psd_matrix(audio_mix_spec, mask1)
        psd_noise = compute_psd_matrix(audio_mix_spec, mask2)
    energy_diagonal_target = np.abs(np.diagonal(psd_target))**2
    np.save('squared_magnitudes_target.npy', energy_diagonal_target)
    energy_diagonal_noise = np.abs(np.diagonal(psd_noise))**2
    np.save('squared_magnitudes_noise.npy', energy_diagonal_noise)
    print(f'Energy diagonal shape before squeeze: {energy_diagonal_target.shape}')  # Print the shape before squeeze
    energy_diagonal_target = np.squeeze(energy_diagonal_target)  # Remove singleton dimensions
    print(f'Energy diagonal shape after squeeze: {energy_diagonal_target.shape}')  # Print the shape after squeeze
    processor.save_results()

if __name__ == '__main__':
    main()