import os
import subprocess
import math
from pathlib import Path
from tqdm import tqdm
import argparse

def get_filelist(root_dir):
    root_path = Path(root_dir)
    fids = []
    for split in ['dev', 'test']:
        all_fns = root_path.glob(f"{split}/mp4/*/*/*.mp4")
        for fn in all_fns:
            relative_fn = fn.relative_to(root_path)
            fids.append(str(relative_fn.with_suffix('')))
    output_fn = root_path / 'file.list'
    with output_fn.open('w') as fo:
        fo.write('\n'.join(fids) + '\n')

def prep_wav(root_dir, wav_dir, flist, ffmpeg, rank, nshard):
    root_path = Path(root_dir)
    print(root_path)
    wav_path = Path(wav_dir)
    os.makedirs(wav_path, exist_ok=True)
    fids = []

    with open(flist, 'r') as file:
        for line in file:  # This iterates over each line in the file
            if not line.strip():  # Skip empty lines
                continue
            corrected_path = Path(line.strip())  # Pathlib automatically corrects the path
            fids.append(corrected_path)

    for fid in fids:
        print(fid)

    num_per_shard = math.ceil(len(fids) / nshard)
    start_id, end_id = num_per_shard * rank, num_per_shard * (rank + 1)
    fids = fids[start_id:end_id]

    print(f"{len(fids)} videos")
    for fid in tqdm(fids):
        video_fn = fid.with_suffix('.mp4')
        print(video_fn)
        audio_fn = wav_path / fid.with_suffix('.wav')
        os.makedirs(audio_fn.parent, exist_ok=True)
        cmd = f'ffmpeg -i "{video_fn}"  -ac 1 "{audio_fn}" -y'
        #cmd = f"{ffmpeg} -i {video_fn} -f wav -vn -y {audio_fn} -loglevel quiet"
        subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VoxCeleb2 data preparation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vox', type=str, required=True, help='VoxCeleb2 dir')
    parser.add_argument('--ffmpeg', type=str, required=True, help='ffmpeg path')
    parser.add_argument('--step', type=int, required=True, help='Steps(1: get file list, 2: extract audio)')
    parser.add_argument('--rank', type=int, default=0, help='rank id')
    parser.add_argument('--nshard', type=int, default=1, help='number of shards')
    args = parser.parse_args()

    if args.step == 1:
        print("Get file list")
        get_filelist(args.vox)
    elif args.step == 2:
        print("Extract audio")
        output_dir = Path(args.vox) / 'audio'
        manifest = Path(args.vox) / 'file.list'
        print('---------------------------------------')
        print(manifest)
        print('---------------------------------------')
        prep_wav(args.vox, output_dir, manifest, args.ffmpeg, args.rank, args.nshard)
