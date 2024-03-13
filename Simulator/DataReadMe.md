" VoxCeleb2 Dataset Preparation Instructions
" ==========================================
Download and decompress the data. Assume the data directory is ${vox}, which contains the folder test. Follow the steps below:

1. Data preparation
python vox_prepare.py --root ${vox} --ffmpeg /path/to/ffmpeg --rank ${rank} --nshard ${nshard} --step ${step}
This will generate a list of file-ids (${vox}/file.list, by step 1) and extract audio wavform from original videos (by step 2). ${step} ranges from 1,2. ${nshard} and ${rank} are only used in step 2. This would shard all videos into ${nshard} and extract audio for ${rank}-th shard, where rank is an integer in [0,nshard-1].
" Step 1: Extract WAV Files
" -------------------------
" To begin processing the VoxCeleb2 dataset, the first step is to extract WAV files from the dataset. 
" Use the following command to perform this action:
:python vox_prepare.py --vox ./data/vox2_test/ --step 2 --ffmpeg ffmpeg --rank 0 --nshard 1

" Step 2: Download Mouth ROIs
" ----------------------------
" The authors of VisualVoice offer pre-processed Mouth Regions of Interest (ROIs) for the VoxCeleb2 dataset. 
" These can be downloaded using the links below:

" - Mouth ROIs for VoxCeleb2 Training Set (1T):
http://dl.fbaipublicfiles.com/VisualVoice/mouth_roi_train.tar.gz

" - Mouth ROIs for VoxCeleb2 Validation Set (20G):
http://dl.fbaipublicfiles.com/VisualVoice/mouth_roi_val.tar.gz

" - Mouth ROIs for Seen-Heard Test Set (88G):
http://dl.fbaipublicfiles.com/VisualVoice/mouth_roi_seen_heard_test.tar.gz

" - Mouth ROIs for Unseen-Unheard Test Set (20G):
http://dl.fbaipublicfiles.com/VisualVoice/mouth_roi_unseen_unheard_test.tar.gz

