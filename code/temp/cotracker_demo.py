import torch
import imageio.v3 as iio
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Download the video
url = 'https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/apple.mp4'

frames = iio.imread(url, plugin="FFMPEG")  # plugin="pyav"

grid_size = 32
window_len = 60
checkpoint = "/workspace/co-tracker/checkpoints/scaled_offline.pth"

video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

# Run Offline CoTracker:
#cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
cotracker = CoTrackerPredictor(
                checkpoint=checkpoint,
                v2=False,
                offline=True,
                window_len=window_len,
            ).to(device)
pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1


vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
vis.visualize(video, pred_tracks, pred_visibility)
