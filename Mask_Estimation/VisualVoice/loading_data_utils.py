import os
import argparse
import soundfile as sf
from scipy.io import wavfile
import numpy as np
import torchvision.transforms as transforms
import torch
#from ..beamformer.GEV_Beamformer import beamformer
from options.test_options import TestOptions
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data.audioVisual_dataset import (generate_spectrogram_complex, load_mouthroi, get_preprocessing_pipelines,
                                      load_frame)
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
    print(f'spec_mix shape is: {spec_mix.shape}')

    if opt.mask_to_use == 'pred':
        mask_prediction_1 = outputs['mask_predictions_A1']
        mask_prediction_2 = outputs['mask_predictions_B1']
        print(f"Shape of mask_prediction_1 before compression: {mask_prediction_1.shape}")
        print(f"Shape of mask_prediction_2 before compression: {mask_prediction_2.shape}")
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
            print(f"Shape of mask_prediction_1 before compression: {mask_prediction_1.shape}")
            print(f"Shape of mask_prediction_2 before compression: {mask_prediction_2.shape}")
    elif opt.mask_to_use == 'gt':
        mask_prediction_1 = outputs['gt_masks_A1'][:, :, :-1, :]
        mask_prediction_2 = outputs['gt_masks_B1'][:, :, :-1, :]
        print(f"Shape of mask_prediction_1 before compression: {mask_prediction_1.shape}")
        print(f"Shape of mask_prediction_2 before compression: {mask_prediction_2.shape}")

    mask_prediction_1.clamp_(-opt.mask_clip_threshold, opt.mask_clip_threshold)
    mask_prediction_2.clamp_(-opt.mask_clip_threshold, opt.mask_clip_threshold)

    spec_mix = spec_mix.numpy()
    print('-------\n-------\n------\n-------\n')
    print(spec_mix.shape)

    pred_masks_1 = mask_prediction_1.detach().cpu().numpy()
    pred_masks_2 = mask_prediction_2.detach().cpu().numpy()
    print(f'pred_masks_1 shape is: {pred_masks_1.shape}')
    pred_spec_1_real = spec_mix[0, 0, :-1] * pred_masks_1[0, 0] - spec_mix[0, 1, :-1] * pred_masks_1[0, 1]
    pred_spec_1_imag = spec_mix[0, 1, :-1] * pred_masks_1[0, 0] + spec_mix[0, 0, :-1] * pred_masks_1[0, 1]
    pred_spec_2_real = spec_mix[0, 0, :-1] * pred_masks_2[0, 0] - spec_mix[0, 1, :-1] * pred_masks_2[0, 1]
    pred_spec_2_imag = spec_mix[0, 1, :-1] * pred_masks_2[0, 0] + spec_mix[0, 0, :-1] * pred_masks_2[0, 1]
    pred_spec_1_real = np.concatenate((pred_spec_1_real, spec_mix[0, 0, -1:, :]), axis=0)
    np.save('model_results/spectrograms/spec_mix.npy', spec_mix)
    pred_spec_1_imag = np.concatenate((pred_spec_1_imag, spec_mix[0, 1, -1:, :]), axis=0)
    pred_spec_2_real = np.concatenate((pred_spec_2_real, spec_mix[0, 0, -1:, :]), axis=0)
    #np.save('model_results/spectrograms/pred_spec_2_real.npy', pred_spec_2_real)
    pred_spec_2_imag = np.concatenate((pred_spec_2_imag, spec_mix[0, 1, -1:, :]), axis=0)
    # Compute the complex spectrograms from the real and imaginary parts
    pred_spec_1_complex = pred_spec_1_real + 1j * pred_spec_1_imag
    pred_spec_2_complex = pred_spec_2_real + 1j * pred_spec_2_imag
    preds_wav_1 = utils.istft_reconstruction_from_complex(pred_spec_1_real, pred_spec_1_imag, hop_length=opt.hop_size,
                                                          length=int(opt.audio_length * opt.audio_sampling_rate))
    preds_wav_2 = utils.istft_reconstruction_from_complex(pred_spec_2_real, pred_spec_2_imag, hop_length=opt.hop_size,
                                                          length=int(opt.audio_length * opt.audio_sampling_rate))
    # Convert the complex mixed spectrogram to magnitude spectrogram
    magnitude_spec_mix = np.abs(spec_mix[0, 0] + 1j * spec_mix[0, 1])

    # Save the magnitude spectrogram of the mixture to an .npy file
    np.save(os.path.join('model_results', 'spectrograms', 'magnitude_spec_mix.npy'), magnitude_spec_mix)
    # Preparing masks as NumPy arrays for return
    output_masks_np = (pred_masks_1, pred_masks_2)
    print('Youhouhou')
    print(f'The mask shape is: {pred_masks_1}')
    # Print to check if the masks are complex numbers

    return (preds_wav_1, preds_wav_2,
            output_masks_np[0], output_masks_np[1],
            pred_spec_1_complex, pred_spec_2_complex)




def initialize_models_and_utilities(opt):
    builder = ModelBuilder()

    # Initialize models
    net_lipreading = builder.build_lipreadingnet(
        config_path=opt.lipreading_config_path,
        weights=opt.weights_lipreadingnet,
        extract_feats=opt.lipreading_extract_feature)
    # if identity feature dim is not 512, for resnet reduce dimension to this feature dim
    if opt.identity_feature_dim != 512:
        opt.with_fc = True
    else:
        opt.with_fc = False

    net_facial_attributes = builder.build_facial(
        pool_type=opt.visual_pool,
        fc_out=opt.identity_feature_dim,
        with_fc=opt.with_fc,
        weights=opt.weights_facial)

    net_unet = builder.build_unet(
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
        audioVisual_feature_dim=opt.audioVisual_feature_dim,
        identity_feature_dim=opt.identity_feature_dim,
        weights=opt.weights_unet)

    net_vocal_attributes = builder.build_vocal(
        pool_type=opt.audio_pool,
        input_channel=2,
        with_fc=opt.with_fc,
        fc_out=opt.identity_feature_dim,
        weights=opt.weights_vocal)

    nets = (net_lipreading, net_facial_attributes, net_unet, net_vocal_attributes)
    model = AudioVisualModel(nets, opt)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    model.to(opt.device)
    model.eval()

    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=True, device=opt.device)

    # Set up preprocessing pipelines
    lipreading_preprocessing_func = get_preprocessing_pipelines()['test']
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    vision_transform_list = [transforms.ToTensor()]
    if opt.normalization:
        vision_transform_list.append(normalize)
    vision_transform = transforms.Compose(vision_transform_list)

    return model, mtcnn, lipreading_preprocessing_func, vision_transform

def load_and_preprocess_data(audio1_path, audio2_path, video1_path, video2_path, mouthroi1_path, mouthroi2_path, mtcnn, vision_transform, opt):
    # Audio loading and normalization
    _, audio1 = wavfile.read(audio1_path)
    _, audio2 = wavfile.read(audio2_path)
    audio1 = audio1 / 32768
    audio2 = audio2 / 32768

    audio_length = min(len(audio1), len(audio2))
    audio1 = clip_audio(audio1[:audio_length])
    audio2 = clip_audio(audio2[:audio_length])
    audio_mix = (audio1 + audio2) / 2.0

    # Decision logic for using MTCNN and loading frames
    if opt.reliable_face:
        best_score_1 = 0
        best_score_2 = 0
        for i in range(10):
            frame_1 = load_frame(video1_path)
            frame_2 = load_frame(video2_path)
            boxes, scores = mtcnn.detect(frame_1)
            if scores and scores[0] > best_score_1:
                best_frame_1 = frame_1
                best_score_1 = scores[0]
            boxes, scores = mtcnn.detect(frame_2)
            if scores and scores[0] > best_score_2:
                best_frame_2 = frame_2
                best_score_2 = scores[0]
        frames_1 = vision_transform(best_frame_1).unsqueeze(0)
        frames_2 = vision_transform(best_frame_2).unsqueeze(0)
    else:
        frame_1_list = []
        frame_2_list = []
        for i in range(opt.number_of_identity_frames):
            frame_1 = load_frame(video1_path)
            frame_2 = load_frame(video2_path)
            frame_1 = vision_transform(frame_1)
            frame_2 = vision_transform(frame_2)
            frame_1_list.append(frame_1)
            frame_2_list.append(frame_2)
        frames_1 = torch.stack(frame_1_list).unsqueeze(0)
        frames_2 = torch.stack(frame_2_list).unsqueeze(0)

    # Loading Mouth ROI Data
    mouthroi_1 = load_mouthroi(mouthroi1_path)
    mouthroi_2 = load_mouthroi(mouthroi2_path)

    return audio1, audio2, audio_mix, audio_length, mouthroi_1, mouthroi_2, frames_1, frames_2

class AudioProcessor:
    def __init__(self, options, beamformer, audio1_path, audio2_path, video1_path, video2_path, mouthroi1_path,
                 mouthroi2_path):
        self.opt = options
        self.beamformer = beamformer
        self.opt.device = torch.device("cuda")

        # Initialize models and utilities
        self.model, self.mtcnn, self.lipreading_preprocessing_func, self.vision_transform = initialize_models_and_utilities(self.opt)

        # Load and preprocess data
        (self.audio1, self.audio2, self.audio_mix, self.audio_length,
         self.mouthroi_1, self.mouthroi_2, self.frames_1, self.frames_2) = load_and_preprocess_data(
            audio1_path, audio2_path, video1_path, video2_path,
            mouthroi1_path, mouthroi2_path, self.mtcnn,
            self.vision_transform, self.opt)

        self.sep_audio1 = np.zeros((self.audio_length))
        self.sep_audio2 = np.zeros((self.audio_length))
        self.overlap_count = np.zeros((self.audio_length))
        self.masks = []  # Initialize an empty list to store masks
        self.masks1 = []  # Initialize an empty list to store masks for audio1
        self.masks2 = []

    def process_audio_segment(self, start, end):
        segment1_audio = self.audio1[start:end]
        segment2_audio = self.audio2[start:end]


        if self.opt.audio_normalization:
            _, segment1_audio = audio_normalize(segment1_audio)
            _, segment2_audio = audio_normalize(segment2_audio)

        audio_mix_spec = generate_spectrogram_complex((segment1_audio + segment2_audio) / 2, self.opt.window_size,
                                                      self.opt.hop_size, self.opt.n_fft)
        audio_spec_1 = generate_spectrogram_complex(segment1_audio, self.opt.window_size, self.opt.hop_size,
                                                    self.opt.n_fft)
        audio_spec_2 = generate_spectrogram_complex(segment2_audio, self.opt.window_size, self.opt.hop_size,
                                                    self.opt.n_fft)

        frame_index_start = int(round(start / self.opt.audio_sampling_rate * 25))
        segment1_mouthroi = self.mouthroi_1[frame_index_start:(frame_index_start + self.opt.num_frames), :, :]
        segment2_mouthroi = self.mouthroi_2[frame_index_start:(frame_index_start + self.opt.num_frames), :, :]

        segment1_mouthroi = self.lipreading_preprocessing_func(segment1_mouthroi)
        segment2_mouthroi = self.lipreading_preprocessing_func(segment2_mouthroi)

        # Properly setting up the data dictionary
        data = {
            'audio_spec_mix1': torch.FloatTensor(audio_mix_spec).unsqueeze(0),
            'mouthroi_A1': torch.FloatTensor(segment1_mouthroi).unsqueeze(0).unsqueeze(0),
            'mouthroi_B': torch.FloatTensor(segment2_mouthroi).unsqueeze(0).unsqueeze(0),
            'audio_spec_A1': torch.FloatTensor(audio_spec_1).unsqueeze(0),
            'audio_spec_B': torch.FloatTensor(audio_spec_2).unsqueeze(0),
            'frame_A': self.frames_1.squeeze(1),  # Ensure this is set only once correctly
            'frame_B': self.frames_2.squeeze(1),
            'mouthroi_A2': torch.FloatTensor(segment1_mouthroi).unsqueeze(0).unsqueeze(0),
            'audio_spec_A2': torch.FloatTensor(audio_spec_1).unsqueeze(0),
            'audio_spec_mix2': torch.FloatTensor(audio_mix_spec).unsqueeze(0)
        }
        if self.opt.audio_normalization:
            normalizer1, segment1_audio = audio_normalize(segment1_audio)
            normalizer2, segment2_audio = audio_normalize(segment2_audio)
        else:
            normalizer1 = 1
            normalizer2 = 1
        #Feeding data to the model
        outputs = self.model.forward(data)
        # the output of the model is going to be fed to get_separated_audio
        # For now I want to only use magnitude masks
        (reconstructed_signal_1, reconstructed_signal_2,
         mask1_np, mask2_np,
         spec_1, spec_2) = get_separated_audio(outputs, data, self.opt, wav=False)
        # Calculate PSD matrices and apply beamforming
        #psd_matrix_1, psd_matrix_2 = self.beamformer.calculate_psd_matrices(audio_mix_spec, mask1_np, mask1_np)
        reconstructed_signal_1 = reconstructed_signal_1 * normalizer1
        reconstructed_signal_2 = reconstructed_signal_2 * normalizer2

        self.sep_audio1[start:end] += reconstructed_signal_1
        self.sep_audio2[start:end] += reconstructed_signal_2
        self.overlap_count[start:end] += 1

        return mask1_np, mask2_np, audio_mix_spec, spec_1, spec_2

    def process_final_segment(self):
        samples_per_window = int(self.opt.audio_length * self.opt.audio_sampling_rate)
        segment1_audio = self.audio1[-samples_per_window:]
        segment2_audio = self.audio2[-samples_per_window:]

        if self.opt.audio_normalization:
            normalizer1, segment1_audio = audio_normalize(segment1_audio)
            normalizer2, segment2_audio = audio_normalize(segment2_audio)
        else:
            normalizer1 = 1
            normalizer2 = 1

        audio_segment = (segment1_audio + segment2_audio) / 2
        audio_mix_spec = generate_spectrogram_complex(audio_segment, self.opt.window_size, self.opt.hop_size,
                                                      self.opt.n_fft)
        audio_spec_1 = generate_spectrogram_complex(segment1_audio, self.opt.window_size, self.opt.hop_size,
                                                    self.opt.n_fft)
        audio_spec_2 = generate_spectrogram_complex(segment2_audio, self.opt.window_size, self.opt.hop_size,
                                                    self.opt.n_fft)

        frame_index_start = int(round((len(self.audio1) - samples_per_window) / self.opt.audio_sampling_rate * 25)) - 1
        segment1_mouthroi = self.mouthroi_1[frame_index_start:(frame_index_start + self.opt.num_frames), :, :]
        segment2_mouthroi = self.mouthroi_2[frame_index_start:(frame_index_start + self.opt.num_frames), :, :]

        segment1_mouthroi = self.lipreading_preprocessing_func(segment1_mouthroi)
        segment2_mouthroi = self.lipreading_preprocessing_func(segment2_mouthroi)

        data = {
            'audio_spec_mix1': torch.FloatTensor(audio_mix_spec).unsqueeze(0),
            'mouthroi_A1': torch.FloatTensor(segment1_mouthroi).unsqueeze(0).unsqueeze(0),
            'mouthroi_B': torch.FloatTensor(segment2_mouthroi).unsqueeze(0).unsqueeze(0),
            'audio_spec_A1': torch.FloatTensor(audio_spec_1).unsqueeze(0),
            'audio_spec_B': torch.FloatTensor(audio_spec_2).unsqueeze(0),
            'frame_A': self.frames_1.squeeze(1),
            'frame_B': self.frames_2.squeeze(1),
            'mouthroi_A2': torch.FloatTensor(segment1_mouthroi).unsqueeze(0).unsqueeze(0),
            'audio_spec_A2': torch.FloatTensor(audio_spec_1).unsqueeze(0),
            'audio_spec_mix2': torch.FloatTensor(audio_mix_spec).unsqueeze(0)
        }

        print('------------------------------------------------')
        print('Now processing the last segment')
        print('------------------------------------------------')
        outputs = self.model.forward(data)
        reconstructed_signal_1, reconstructed_signal_2, mask1_np, mask2_np = get_separated_audio(outputs, data,
                                                                                                 self.opt)
        reconstructed_signal_1 = reconstructed_signal_1 * normalizer1
        reconstructed_signal_2 = reconstructed_signal_2 * normalizer2
        self.sep_audio1[-samples_per_window:] += reconstructed_signal_1
        self.sep_audio2[-samples_per_window:] += reconstructed_signal_2
        self.overlap_count[-samples_per_window:] += 1
        return mask1_np, mask2_np, audio_mix_spec


    def combine_masks(self, overlap=None):
        # If non-overlapping, concatenate masks
        if not overlap:
            full_mask = np.concatenate(self.masks, axis=-1)
        else:
            # For overlapping segments, average or sum at the overlaps
            full_length = self.audio_length  # Total length of the audio
            full_mask = np.zeros((self.masks[0].shape[0], full_length))  # Initialize a mask of zeros
            coverage = np.zeros(full_length)  # To track coverage for averaging

            position = 0
            for mask in self.masks:
                end_position = position + mask.shape[1]
                full_mask[:, position:end_position] += mask
                coverage[position:end_position] += 1
                position += self.opt.hop_size  # Move position by hop size for next segment

            full_mask /= np.maximum(1, coverage)  # Avoid division by zero

        return full_mask

    def save_results(self, audio1_path, audio2_path, video1_path, video2_path):
        # Extract video names from the path to use in the output directory name
        # Split paths using the os.path.sep which will be '\\' on Windows and '/' on Unix
        parts1 = video1_path.split(os.path.sep)
        parts2 = video2_path.split(os.path.sep)
        # Use os.path.basename to extract file names without the full path
        audio1_basename = os.path.basename(audio1_path).replace('.wav', '')
        audio2_basename = os.path.basename(audio2_path).replace('.wav', '')



        # Check if parts have enough elements before accessing
        if len(parts1) >= 3:
            video1_name = parts1[-3] + '_' + parts1[-2] + '_' + parts1[-1][:-4]
        else:
            video1_name = os.path.basename(video1_path).split('.')[0]

        if len(parts2) >= 3:
            video2_name = parts2[-3] + '_' + parts2[-2] + '_' + parts2[-1][:-4]
        else:
            video2_name = os.path.basename(video2_path).split('.')[0]

        # Create a directory based on the video names
        output_dir = os.path.join(self.opt.output_dir_root, video1_name + 'VS' + video2_name)
        os.makedirs(output_dir, exist_ok=True)

        output_audio1_path = os.path.join(output_dir, f"{audio1_basename}.wav")
        output_audio2_path = os.path.join(output_dir, f"{audio2_basename}.wav")
        output_audio_mixed_path = os.path.join(output_dir, f"{audio1_basename}_and_{audio2_basename}_mixed.wav")
        output_audio1_separated_path = os.path.join(output_dir, f"{audio1_basename}_separated.wav")
        output_audio2_separated_path = os.path.join(output_dir, f"{audio2_basename}_separated.wav")
        output_mask1_path = os.path.join(output_dir, f"{audio1_basename}_mask.npy")
        output_mask2_path = os.path.join(output_dir, f"{audio2_basename}_mask.npy")

        # Save the original and processed audio files
        sf.write(output_audio1_path, self.audio1, self.opt.audio_sampling_rate)
        sf.write(output_audio2_path, self.audio2, self.opt.audio_sampling_rate)
        sf.write(output_audio_mixed_path, self.audio_mix, self.opt.audio_sampling_rate)

        # Calculate the averaged separated audio accounting for overlaps
        avged_sep_audio1 = np.divide(self.sep_audio1, self.overlap_count, out=np.zeros_like(self.sep_audio1),
                                     where=self.overlap_count != 0)
        avged_sep_audio2 = np.divide(self.sep_audio2, self.overlap_count, out=np.zeros_like(self.sep_audio2),
                                     where=self.overlap_count != 0)

        # Save the separated audio files
        sf.write(output_audio1_separated_path, avged_sep_audio1, self.opt.audio_sampling_rate)
        sf.write(output_audio2_separated_path, avged_sep_audio2, self.opt.audio_sampling_rate)

        np.save(output_mask1_path, self.masks1)
        np.save(output_mask2_path, self.masks2)

        print(f'Files saved to {output_dir}')
        return output_dir

    def get_psd_matrix(self):
        psd_matrix_target = self.beamformer.get_psd_matrix(self.spec)
        psd_matrix_noise = self.beamformer.get_psd_matrix(self.spec)
        return psd_matrix_target, psd_matrix_noise

