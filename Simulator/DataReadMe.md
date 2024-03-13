This is just a draft: 
VoxCeleb2 Dataset
first step is to extract the wav files from the dataset using:
python vox_prepare.py --vox ./data/vox2_test/  --step 2 --ffmpeg ffmpeg --rank 0 --nshard 1

