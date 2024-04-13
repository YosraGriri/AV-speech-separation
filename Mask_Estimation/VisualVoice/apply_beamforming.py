from test_utils import*


def main():
    opt = TestOptions().parse()
    processor = AudioProcessor(opt)

    samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
    sliding_window_start = 0
    while sliding_window_start + samples_per_window < processor.audio_length:
        sliding_window_end = sliding_window_start + samples_per_window
        processor.process_audio_segment(sliding_window_start, sliding_window_end)
        sliding_window_start += int(opt.hop_length * opt.audio_sampling_rate)

    processor.process_final_segment()
    processor.save_results()


if __name__ == '__main__':
    main()