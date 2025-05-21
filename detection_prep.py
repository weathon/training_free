import os
from PIL import Image
import imageio
import numpy as np

input_path = "/home/wg25r/fastdata/fullmoca/MoCA-Video-Test/"
videos = sorted(os.listdir(input_path))
output_path = "test"

os.makedirs(output_path, exist_ok=True)
for videos in videos:
    video_path = os.path.join(input_path, videos, "Frame")
    frames = sorted(os.listdir(video_path))
    frames = [frame for frame in frames if int(frame.replace(".jpg", "")) % 5 == 0]
    writer = imageio.get_writer('./test/{}.mp4'.format(videos), fps=30)
    
    for i, frame in enumerate(range(0, min(37, len(frames)))):
        frame_path = os.path.join(video_path, frames[i])
        image = Image.open(frame_path)
        image = image.convert("RGB")
        image = np.array(image)
        writer.append_data(image)
    writer.close()