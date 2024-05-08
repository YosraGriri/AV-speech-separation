import os
import argparse
import soundfile as sf
from scipy.io import wavfile
import numpy as np
import torchvision.transforms as transforms
import torch
from options.test_options import TestOptions
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data.audioVisual_dataset import generate_spectrogram_complex, load_mouthroi, get_preprocessing_pipelines, \
    load_frame
from utils import utils
from utils.lipreading_preprocess import *
from utils.beamformer import calculate_psd_matrices
from shutil import copy
from facenet_pytorch import MTCNN


def audio_normalize(samples, desired_rms=0.1, eps=1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples ** 2)))
    samples = samples * (desired_rms / rms)
    return rms / desired_rms, samples


def clip_audio(audio):
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def print_mask_info(mask, mask_name="Mask"):
    print(f"{mask_name} Info:")
    print("Shape:", mask.shape)
    print("Dtype:", mask.dtype)
    # Check if the tensor is complex
    if torch.is_complex(mask):
        print("Contains complex values.")
        # You can also print some of the values to inspect
        print("Sample values (real part):", mask.real[:5])
        print("Sample values (imaginary part):", mask.imag[:5])
    else:
        print("Contains real values only.")
        # Print first few values
        print("Sample values:", mask[:5])

def get_separated_audio(outputs, batch_data, opt, wav=False):
    """
       Extracts separated audio signals from the mixed audio input based on model predictions.

       This function takes as input the batch data containing the mixed audio spectrogram and
       the prediction outputs from the deep learning model, which include the masks for each audio
       source within the mixed audio. These masks are essentially filters that indicate which parts
       of the mixed spectrogram belong to each of the original audio sources.

       The function applies these predicted masks to the mixed audio spectrogram to isolate the
       spectrograms of the individual sources. It performs this operation by element-wise multiplying
       the mixed spectrogram with each mask, separately for each source. This operation isolates
       components of the mixed signal attributed to each source based on the model's predictions.

       After applying the masks, the function reconstructs the audio signals for each source from
       their isolated spectrograms. It converts these isolated spectrograms back into time-domain
       audio signals using the inverse Short-Time Fourier Transform (iSTFT). This reconstruction
       process creates the separated audio signals from the complex spectrogram representations.

       Parameters:
           outputs (dict): The outputs from the deep learning model, containing the predicted masks
               for audio separation.
           batch_data (dict): The input batch data to the model, containing the mixed audio spectrogram
               among other pieces of data.
           opt (argparse.Namespace): A namespace object containing options/configurations for the
               audio separation process.

       Returns:
           Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays, each representing
               one of the separated audio signals in the time domain.
       """
    # fetch data and predictions
    spec_mix = batch_data['audio_spec_mix1']

    if opt.mask_to_use == 'pred':
        mask_prediction_1 = outputs['mask_predictions_A1']
        mask_prediction_2 = outputs['mask_predictions_B1']
        if opt.compression_type == 'hyperbolic':
            K = opt.hyperbolic_compression_K
            C = opt.hyperbolic_compression_C
            mask_prediction_1 = - torch.log((K - mask_prediction_1) / (K + mask_prediction_1)) / C
            mask_prediction_2 = - torch.log((K - mask_prediction_2) / (K + mask_prediction_2)) / C
        elif opt.compression_type == 'sigmoidal':
            a = opt.sigmoidal_compression_a
            b = opt.sigmoidal_compression_b
            mask_prediction_1 = (b - torch.log(1 / mask_prediction_1 - 1)) / a
            mask_prediction_2 = (b - torch.log(1 / mask_prediction_2 - 1)) / a
    elif opt.mask_to_use == 'gt':
        mask_prediction_1 = outputs['gt_masks_A1'][:, :, :-1, :]
        mask_prediction_2 = outputs['gt_masks_B1'][:, :, :-1, :]

    mask_prediction_1.clamp_(-opt.mask_clip_threshold, opt.mask_clip_threshold)
    mask_prediction_2.clamp_(-opt.mask_clip_threshold, opt.mask_clip_threshold)
    print_mask_info(mask_prediction_1, "Mask Prediction 1")

    spec_mix = spec_mix.numpy()
    pred_masks_1 = mask_prediction_1.detach().cpu().numpy()
    pred_masks_2 = mask_prediction_2.detach().cpu().numpy()

    pred_spec_1_real = spec_mix[0, 0, :-1] * pred_masks_1[0, 0] - spec_mix[0, 1, :-1] * pred_masks_1[0, 1]
    pred_spec_1_imag = spec_mix[0, 1, :-1] * pred_masks_1[0, 0] + spec_mix[0, 0, :-1] * pred_masks_1[0, 1]
    pred_spec_2_real = spec_mix[0, 0, :-1] * pred_masks_2[0, 0] - spec_mix[0, 1, :-1] * pred_masks_2[0, 1]
    pred_spec_2_imag = spec_mix[0, 1, :-1] * pred_masks_2[0, 0] + spec_mix[0, 0, :-1] * pred_masks_2[0, 1]
    pred_spec_1_real = np.concatenate((pred_spec_1_real, spec_mix[0, 0, -1:, :]), axis=0)
    pred_spec_1_imag = np.concatenate((pred_spec_1_imag, spec_mix[0, 1, -1:, :]), axis=0)
    pred_spec_2_real = np.concatenate((pred_spec_2_real, spec_mix[0, 0, -1:, :]), axis=0)
    pred_spec_2_imag = np.concatenate((pred_spec_2_imag, spec_mix[0, 1, -1:, :]), axis=0)

    preds_wav_1 = utils.istft_reconstruction_from_complex(pred_spec_1_real, pred_spec_1_imag, hop_length=opt.hop_size,
                                                          length=int(opt.audio_length * opt.audio_sampling_rate))
    preds_wav_2 = utils.istft_reconstruction_from_complex(pred_spec_2_real, pred_spec_2_imag, hop_length=opt.hop_size,
                                                          length=int(opt.audio_length * opt.audio_sampling_rate))

    # Preparing masks as NumPy arrays for return
    output_masks_np = (mask_prediction_1.detach().cpu().numpy(), mask_prediction_2.detach().cpu().numpy())

    return preds_wav_1, preds_wav_2


def main():
    ##Parse command-line arguments to configure the run.


    opt = TestOptions().parse()
    # Set device to CUDA, indicating the use of GPU.
    opt.device = torch.device("cuda")



    # Network Builders

    # Instantiate a ModelBuilder, responsible for creating model components.
    builder = ModelBuilder()
    # Build various components of the model using configurations and weights specified in `opt`
    net_lipreading = builder.build_lipreadingnet(
        config_path=opt.lipreading_config_path,
        weights=opt.weights_lipreadingnet,
        extract_feats=opt.lipreading_extract_feature)
    # if identity feature dim is not 512, for resnet reduce dimension to this feature dim
    if opt.identity_feature_dim != 512:
        opt.with_fc = True
    else:
        opt.with_fc = False
    """Function: Extracts visual features related to lip movements from video frames. 
	These features are critical for tasks like speech recognition or speaker identification
	from visual data."""

    net_facial_attributes = builder.build_facial(
        pool_type=opt.visual_pool,
        fc_out=opt.identity_feature_dim,
        with_fc=opt.with_fc,
        weights=opt.weights_facial)
    """Function: Extracts features from facial images that are indicative of 
	the person's identity or other attributes. 
	This can complement lipreading features by providing additional context."""

    net_unet = builder.build_unet(
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
        audioVisual_feature_dim=opt.audioVisual_feature_dim,
        identity_feature_dim=opt.identity_feature_dim,
        weights=opt.weights_unet)
    """
    Function: Processes audio signals, likely for tasks such as
	source separation or enhancement, using a U-Net architecture known 
	for its effectiveness in such tasks.
	"""

    net_vocal_attributes = builder.build_vocal(
        pool_type=opt.audio_pool,
        input_channel=2,  # Complex spectrogram Maginitude channel and phase channel, not confuse with stereo!!!
        with_fc=opt.with_fc,
        fc_out=opt.identity_feature_dim,
        weights=opt.weights_vocal)
    """Function: Similar to the facial attributes network but focuses on 
	extracting features from audio signals that reflect the speaker's 
	vocal characteristics."""

    nets = (net_lipreading, net_facial_attributes, net_unet, net_vocal_attributes)
    # print(nets)

    # construct our audio-visual model
    # Assemble the model from its components.
    model = AudioVisualModel(nets, opt)

    # Prepare model for possible multi-GPU execution.
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    # Move model to the GPU.
    model.to(opt.device)

    # Set model to evaluation mode, disabling dropout or batchnorm updates.
    model.eval()

    mtcnn = MTCNN(keep_all=True, device=opt.device)

    lipreading_preprocessing_func = get_preprocessing_pipelines()['test']
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    vision_transform_list = [transforms.ToTensor()]
    if opt.normalization:
        vision_transform_list.append(normalize)
    vision_transform = transforms.Compose(vision_transform_list)

    # load data
    print('Will start loading mouth data')
    mouthroi_1 = load_mouthroi(opt.mouthroi1_path)
    print('mouth roi data is loaded')

    mouthroi_2 = load_mouthroi(opt.mouthroi2_path)

    _, audio1 = wavfile.read(opt.audio1_path)
    _, audio2 = wavfile.read(opt.audio2_path)
    audio1 = audio1 / 32768
    # normalization step -1.0 and 1.0 commonly used in audio processing,
    # especially when dealing with 16-bit PCM audio data.
    audio2 = audio2 / 32768

    # make sure the two audios are of the same length and then mix them
    audio_length = min(len(audio1), len(audio2))
    audio1 = clip_audio(audio1[:audio_length])
    audio2 = clip_audio(audio2[:audio_length])
    audio_mix = (audio1 + audio2) / 2.0  # simple mixing to a mono file by avg the two signals

    if opt.reliable_face:
        best_score_1 = 0
        best_score_2 = 0
        for i in range(10):
            frame_1 = load_frame(opt.video1_path)
            frame_2 = load_frame(opt.video2_path)
            boxes, scores = mtcnn.detect(frame_1)
            if scores[0] > best_score_1:
                best_frame_1 = frame_1
            boxes, scores = mtcnn.detect(frame_2)
            if scores[0] > best_score_2:
                best_frame_2 = frame_2
        frames_1 = vision_transform(best_frame_1).squeeze().unsqueeze(0)
        frames_2 = vision_transform(best_frame_2).squeeze().unsqueeze(0)
    else:
        frame_1_list = []
        frame_2_list = []
        for i in range(opt.number_of_identity_frames):
            frame_1 = load_frame(opt.video1_path)
            frame_2 = load_frame(opt.video2_path)
            frame_1 = vision_transform(frame_1)
            frame_2 = vision_transform(frame_2)
            frame_1_list.append(frame_1)
            frame_2_list.append(frame_2)
        frames_1 = torch.stack(frame_1_list).squeeze().unsqueeze(0)
        frames_2 = torch.stack(frame_2_list).squeeze().unsqueeze(0)
    """Depending on the opt.reliable_face flag, the script either selects the best identity 
    frames using facial detection scores (using MTCNN) or loads a predefined number of frames directly. 
	This is crucial for extracting visual features related to identity."""

    # perform separation over the whole audio using a sliding window approach
    overlap_count = np.zeros((audio_length))
    sep_audio1 = np.zeros((audio_length))
    sep_audio2 = np.zeros((audio_length))
    sliding_window_start = 0
    data = {}
    avged_sep_audio1 = np.zeros((audio_length))
    avged_sep_audio2 = np.zeros((audio_length))

    samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
    while sliding_window_start + samples_per_window < audio_length:
        # Code to process each window
        sliding_window_end = sliding_window_start + samples_per_window

        # get audio spectrogram
        segment1_audio = audio1[sliding_window_start:sliding_window_end]
        segment2_audio = audio2[sliding_window_start:sliding_window_end]

        if opt.audio_normalization:
            normalizer1, segment1_audio = audio_normalize(segment1_audio)
            normalizer2, segment2_audio = audio_normalize(segment2_audio)
        else:
            normalizer1 = 1
            normalizer2 = 1

        audio_segment = (segment1_audio + segment2_audio) / 2
        audio_mix_spec = generate_spectrogram_complex(audio_segment, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_1 = generate_spectrogram_complex(segment1_audio, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_2 = generate_spectrogram_complex(segment2_audio, opt.window_size, opt.hop_size, opt.n_fft)

        # get mouthroi
        frame_index_start = int(round(sliding_window_start / opt.audio_sampling_rate * 25))
        segment1_mouthroi = mouthroi_1[frame_index_start:(frame_index_start + opt.num_frames), :, :]
        segment2_mouthroi = mouthroi_2[frame_index_start:(frame_index_start + opt.num_frames), :, :]

        # transform mouthrois
        segment1_mouthroi = lipreading_preprocessing_func(segment1_mouthroi)
        segment2_mouthroi = lipreading_preprocessing_func(segment2_mouthroi)

        data['audio_spec_mix1'] = torch.FloatTensor(audio_mix_spec).unsqueeze(0)
        data['mouthroi_A1'] = torch.FloatTensor(segment1_mouthroi).unsqueeze(0).unsqueeze(0)
        data['mouthroi_B'] = torch.FloatTensor(segment2_mouthroi).unsqueeze(0).unsqueeze(0)
        data['audio_spec_A1'] = torch.FloatTensor(audio_spec_1).unsqueeze(0)
        data['audio_spec_B'] = torch.FloatTensor(audio_spec_2).unsqueeze(0)
        data['frame_A'] = frames_1
        data['frame_B'] = frames_2
        # don't care below
        data['frame_A'] = frames_1
        data['mouthroi_A2'] = torch.FloatTensor(segment1_mouthroi).unsqueeze(0).unsqueeze(0)
        data['audio_spec_A2'] = torch.FloatTensor(audio_spec_1).unsqueeze(0)
        data['audio_spec_mix2'] = torch.FloatTensor(audio_mix_spec).unsqueeze(0)

        for key, tensor in data.items():
            print(f"Shape of {key} before model call: {tensor.shape}")

        outputs = model.forward(data)
        print('Hellloo finished processing most of the signal! ')
        print('Now Getting the masks')
        reconstructed_signal_1, reconstructed_signal_2 = get_separated_audio(outputs, data, opt,
                                                                                     wav=False)
        base_id1 = os.path.splitext(os.path.basename(opt.audio1_path))[0]
        base_id2 = os.path.splitext(os.path.basename(opt.audio2_path))[0]
        print(f'First baseID: {base_id1}\n'
              f'second baseID: {base_id2}')
        # Generate a base filename for the masks
        mask_filename_base = utils.generate_mask_filename(base_id1, base_id2).split('_mask.npy')[
            0]  # To match with the expected input for 'save_masks_as_numpy'
        output_dir_base = "../../data/Masks"
        output_dir = os.path.join(output_dir_base, mask_filename_base)
        #utils.save_masks_as_numpy([mask1_np, mask2_np], base_id1, base_id2, output_dir)
        reconstructed_signal_1 = reconstructed_signal_1 * normalizer1
        reconstructed_signal_2 = reconstructed_signal_2 * normalizer2
        sep_audio1[sliding_window_start:sliding_window_end] = sep_audio1[
                                                              sliding_window_start:sliding_window_end] + reconstructed_signal_1
        sep_audio2[sliding_window_start:sliding_window_end] = sep_audio2[
                                                              sliding_window_start:sliding_window_end] + reconstructed_signal_2
        # update overlap count
        overlap_count[sliding_window_start:sliding_window_end] = overlap_count[
                                                                 sliding_window_start:sliding_window_end] + 1
        sliding_window_start = sliding_window_start + int(opt.hop_length * opt.audio_sampling_rate)

        # deal with the last segment
        """The explicit handling of the last audio segment outside the main loop 
        ensures that any remaining audio 
        (that didn't fit into the sliding windows due to length mismatches) 
        is also processed. 
        This is a standard approach to ensure full coverage of the audio file."""

    # get audio spectrogram
    segment1_audio = audio1[-samples_per_window:]
    segment2_audio = audio2[-samples_per_window:]

    if opt.audio_normalization:
        normalizer1, segment1_audio = audio_normalize(segment1_audio)
        normalizer2, segment2_audio = audio_normalize(segment2_audio)
    else:
        normalizer1 = 1
        normalizer2 = 1

    audio_segment = (segment1_audio + segment2_audio) / 2
    audio_mix_spec = generate_spectrogram_complex(audio_segment, opt.window_size, opt.hop_size, opt.n_fft)
    # get mouthroi
    frame_index_start = int(round((len(audio1) - samples_per_window) / opt.audio_sampling_rate * 25)) - 1
    segment1_mouthroi = mouthroi_1[frame_index_start:(frame_index_start + opt.num_frames), :, :]
    segment2_mouthroi = mouthroi_2[frame_index_start:(frame_index_start + opt.num_frames), :, :]
    # transform mouthrois
    segment1_mouthroi = lipreading_preprocessing_func(segment1_mouthroi)
    segment2_mouthroi = lipreading_preprocessing_func(segment2_mouthroi)
    audio_spec_1 = generate_spectrogram_complex(segment1_audio, opt.window_size, opt.hop_size, opt.n_fft)
    audio_spec_2 = generate_spectrogram_complex(segment2_audio, opt.window_size, opt.hop_size, opt.n_fft)
    data['audio_spec_mix1'] = torch.FloatTensor(audio_mix_spec).unsqueeze(0)
    data['mouthroi_A1'] = torch.FloatTensor(segment1_mouthroi).unsqueeze(0).unsqueeze(0)
    data['mouthroi_B'] = torch.FloatTensor(segment2_mouthroi).unsqueeze(0).unsqueeze(0)
    data['audio_spec_A1'] = torch.FloatTensor(audio_spec_1).unsqueeze(0)
    data['audio_spec_B'] = torch.FloatTensor(audio_spec_2).unsqueeze(0)
    data['frame_A'] = frames_1
    data['frame_B'] = frames_2
    # don't care below
    data['frame_A'] = frames_1
    data['mouthroi_A2'] = torch.FloatTensor(segment1_mouthroi).unsqueeze(0).unsqueeze(0)
    data['audio_spec_A2'] = torch.FloatTensor(audio_spec_1).unsqueeze(0)
    data['audio_spec_mix2'] = torch.FloatTensor(audio_mix_spec).unsqueeze(0)
    print('------------------------------------------------')
    print('now let us process the last segment')
    print('------------------------------------------------')
    outputs = model.forward(data)
    reconstructed_signal_1, reconstructed_signal_2 = get_separated_audio(outputs, data, opt)
    reconstructed_signal_1 = reconstructed_signal_1 * normalizer1
    reconstructed_signal_2 = reconstructed_signal_2 * normalizer2
    sep_audio1[-samples_per_window:] = sep_audio1[-samples_per_window:] + reconstructed_signal_1
    sep_audio2[-samples_per_window:] = sep_audio2[-samples_per_window:] + reconstructed_signal_2
    # update overlap count
    overlap_count[-samples_per_window:] = overlap_count[-samples_per_window:] + 1

    # divide the aggregated predicted audio by the overlap count
    avged_sep_audio1 = avged_sep_audio1 + clip_audio(np.divide(sep_audio1, overlap_count))
    avged_sep_audio2 = avged_sep_audio2 + clip_audio(np.divide(sep_audio2, overlap_count))

    # output original and separated audios
    parts1 = opt.video1_path.split('/')
    parts2 = opt.video2_path.split('/')
    video1_name = parts1[-3] + '_' + parts1[-2] + '_' + parts1[-1][:-4]
    video2_name = parts2[-3] + '_' + parts2[-2] + '_' + parts2[-1][:-4]

    output_dir = os.path.join(opt.output_dir_root, video1_name + 'VS' + video2_name)
    os.makedirs(output_dir, exist_ok=True)
    sf.write(os.path.join(output_dir, 'audio1.wav'), audio1, opt.audio_sampling_rate)
    print(f'saving file to {output_dir}')

    sf.write(os.path.join(output_dir, 'audio2.wav'), audio2, opt.audio_sampling_rate)
    sf.write(os.path.join(output_dir, 'audio_mixed.wav'), audio_mix, opt.audio_sampling_rate)
    sf.write(os.path.join(output_dir, 'audio1_separated.wav'), avged_sep_audio1, opt.audio_sampling_rate)
    sf.write(os.path.join(output_dir, 'audio2_separated.wav'), avged_sep_audio2, opt.audio_sampling_rate)


if __name__ == '__main__':
    main()
