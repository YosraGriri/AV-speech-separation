from generate_sample import *
import multiprocessing.dummy as mp
from generate_sample_all import generate_room_simulations

def main():
    parser = argparse.ArgumentParser(description='Generate room acoustics samples.')
    parser.add_argument('--output_path', type=str,
                        help='Output path for saving samples', required=True)
    parser.add_argument('--source_dir', type=str,
                        help='path containing sources', required=True)
    parser.add_argument('--voice_identity', type=int, default=0,
                        help='For the position of the voice: voice_identity*angle')
    parser.add_argument('--max_order', type=int, default=4,
                        help='Output path for saving room')
    parser.add_argument('--dataset', type=str, default='',
                        help='which dataset to process')
    parser.add_argument('--voice_theta', type=float, default=2.55,
                        help='Duration of the generated samples in seconds')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sampling rate of the generated samples')
    parser.add_argument('--random', type=bool, default=False,
                        help='Sampling rate of the generated samples')
    parser.add_argument('--n_sources', type=int, default=2,
                        help='Number of sound sources in the room')
    parser.add_argument('--n_mics', type=int, default=4,
                        help='Number of microphones in the room')
    parser.add_argument('--n_outputs', type=int, default=2,
                        help='Number of outputs')
    parser.add_argument('--n_samples', type=int, default=1,
                        help='Number of outputs')
    parser.add_argument('--mic_center', type=list, default=[0, 0, 1.7],
                        help='Number of microphones in the room')
    parser.add_argument("--angle_between_sources", type=float, default=15.0,
                        help="Angle between voice sources in degrees.")
    parser.add_argument('--mic_radius', type=float, default=0.035,
                        help='Radius of the circular microphone array')
    parser.add_argument('--voice_radius', type=list, default=[1.0, 2.0],
                        help='Range of voice source radius')
    parser.add_argument('--rt60', type=float, default=0.3,
                        help='Reverberation time of the room')
    parser.add_argument('--room_dimensions', type=float, nargs=3, default=[3.5, 3.5, 2.5],
                        help='Room dimensions (length, width, height)')
    parser.add_argument('--avg_human_height', type=float, default=1.7,
                        help='Average human height')

    args = parser.parse_args()
    generate_room_simulations(args)
    #generate_unique_samples(args, args.n_samples)
    #generate_samples_for_all_folders(args)

if __name__ == "__main__":
    main()