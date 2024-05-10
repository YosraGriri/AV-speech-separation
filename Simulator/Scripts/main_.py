from generate_sample import generate_unique_samples
from create_room_simulations import create_room_simulations
from collect_voice_metadata import main_collect_voice_metadata
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate room acoustics samples including voice and microphone positions with specified room dimensions and acoustics properties.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for saving generated acoustic samples.')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Path containing source audio files.')
    parser.add_argument('--max_order', type=int, default=0,
                        help='Maximum order of reflection to simulate.')
    parser.add_argument('--dataset', type=str, default='',
                        help='Specifies which dataset to process.')
    parser.add_argument('--duration', type=float,
                        help='Duration of the generated samples in seconds.')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sampling rate of the generated samples.')
    parser.add_argument('--n_sources', type=int, default=2,
                        help='Number of sound sources to simulate in the room.')
    parser.add_argument('--n_mics', type=int, default=4,
                        help='Number of microphones in the simulation.')
    parser.add_argument('--n_outputs', type=int, default=2,
                        help='Number of audio output files to generate.')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Total number of samples to generate.')
    parser.add_argument('--mic_center', type=float, nargs=3, default=[0, 0, 1.7],
                        help='Center position of the microphone array (x, y, z).')
    parser.add_argument("--angle_between_sources", type=float, default=180.0,
                        help="Angle in degrees between multiple sound sources.")
    parser.add_argument('--mic_radius', type=float, default=0.035,
                        help='Radius of the circular microphone array in meters.')
    parser.add_argument('--voice_radius', type=float, nargs=2, default=1.7,
                        help='Minimum and maximum radii for positioning voice sources from the center, in meters.')
    parser.add_argument('--rt60', type=float, default=0.0,
                        help='Target reverberation time (RT60) of the room in seconds.')
    parser.add_argument('--room_dimensions', type=float, nargs=3, default=[3.5, 3.5, 2.5],
                        help='Room dimensions as a list: length, width, height in meters.')
    parser.add_argument('--avg_human_height', type=float, default=1.7,
                        help='Average height of a human in meters, used for scaling.')
    parser.add_argument('--random', type=bool, default=False,
                        help='Boolean flag to randomize the positions of the sources.')
    parser.add_argument('--save-plot', action='store_true',
                        help='Flag to save a plot of the room setup and source positions. No value needed.')
    parser.add_argument('--voice_theta', type=float, default=90,
                        help='Polar angle for positioning voice sources in degrees.')

    args = parser.parse_args()
    #for now because of memory issue on laptop only the first 175 samples
    # from the list are to be processed and tested with
    create_room_simulations(args)
if __name__ == "__main__":
    main()
