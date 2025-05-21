# %% [markdown]
# 

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from diffusers import AutoencoderKLMochi
from mochi_pipeline import MochiPipeline, linear_quadratic_schedule, retrieve_timesteps
from transformers import T5EncoderModel, T5Tokenizer
from mochi_processor import MochiAttnProcessor2_0
import torch
import torchvision
from pathlib import Path
from diffusers.utils import export_to_video

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, default="./cog_dataset/videos/spider_tailed_horned_viper_1_001.mp4")
parser.add_argument("--run_id", type=str)

args = parser.parse_args()


video_path = args.video_path
import wandb
wandb.init(project="mochi", resume=True, id=args.run_id)

def encode_videos(model: torch.nn.Module, vid_path: Path):
    video, _, metadata = torchvision.io.read_video(str(vid_path), output_format="THWC", pts_unit="secs")
    video = video.permute(3, 0, 1, 2)
    og_shape = video.shape
    
    print(f"Original video shape: {og_shape}")
    video = video.unsqueeze(0)
    video = video.float() / 127.5 - 1.0
    video = video.to(model.device)

    assert video.ndim == 5

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ldist = model._encode(video)
    return ldist


# %%


# %%
model_id = "genmo/mochi-1-preview"
pipeline = MochiPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

pipeline.enable_vae_slicing()
pipeline.enable_vae_tiling()
pipeline.to("cuda")

# %%
with torch.inference_mode():
    video = video_path
    latents = encode_videos(
        pipeline.vae,
        Path(video),
    )[:,:12].to(pipeline.vae.device).to(pipeline.vae.dtype) 

# %%
pipeline.enable_vae_slicing()
pipeline.enable_vae_tiling()
pipeline.enable_model_cpu_offload()
from diffusers.utils import export_to_video


with torch.no_grad():
    has_latents_mean = hasattr(pipeline.vae.config, "latents_mean") and pipeline.vae.config.latents_mean is not None
    has_latents_std = hasattr(pipeline.vae.config, "latents_std") and pipeline.vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = (
            torch.tensor(pipeline.vae.config.latents_mean).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(pipeline.vae.config.latents_std).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
        )
        latents_ = latents * latents_std / pipeline.vae.config.scaling_factor + latents_mean
    else:
        latents_ = latents / pipeline.vae.config.scaling_factor

    video = pipeline.vae.decode(latents_, return_dict=False)[0]
    video = pipeline.video_processor.postprocess_video(video, output_type="pil")

export_to_video(
    video[0],
    "output.mp4",
    fps=30
)

# %%
import numpy as np
tokenizer = pipeline.tokenizer
prompt = "a camflagued *animal* moving in the +background+"
positive_start_index = tokenizer.tokenize(prompt).index("▁*")
positive_end_index = tokenizer.tokenize(prompt).index("*")

negative_start_index = tokenizer.tokenize(prompt).index("▁+")
negative_end_index = tokenizer.tokenize(prompt).index("+")

print(tokenizer.tokenize(prompt))


indices = torch.tensor(list(range(positive_start_index + 1, positive_end_index)) + list(range(negative_start_index + 1, negative_end_index)))# + 1 # plus one because of the <extra_id_0> token
print(indices)
positive_mask = torch.tensor([1] * (positive_end_index - positive_start_index - 1) + [0] * (negative_end_index - negative_start_index - 1))
negative_mask = torch.tensor([0] * (positive_end_index - positive_start_index - 1) + [1] * (negative_end_index - negative_start_index - 1))
print(positive_mask)
print(negative_mask)

threshold_noise = 0.025
sigmas = linear_quadratic_schedule(20, threshold_noise)
sigmas = np.array(sigmas)
timesteps = retrieve_timesteps(
            pipeline.scheduler,
            num_inference_steps=20,
            device="cuda",
            timesteps=None,
            sigmas=sigmas,
        )

for block in pipeline.transformer.transformer_blocks:
    block.attn1.processor = MochiAttnProcessor2_0(token_index_of_interest=indices, positive_mask=positive_mask) 

# %%
noise = torch.randn_like(latents)
latents = (latents - latents_mean) / latents_std 
noisy_latents = pipeline.scheduler.scale_noise(latents, timesteps[0][15:16], noise) 

# %%
# pipeline.scheduler.num_inference_steps

# %%
# with torch.no_grad():
#     has_latents_mean = hasattr(pipeline.vae.config, "latents_mean") and pipeline.vae.config.latents_mean is not None
#     has_latents_std = hasattr(pipeline.vae.config, "latents_std") and pipeline.vae.config.latents_std is not None
#     if has_latents_mean and has_latents_std:
#         latents_mean = (
#             torch.tensor(pipeline.vae.config.latents_mean).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
#         )
#         latents_std = (
#             torch.tensor(pipeline.vae.config.latents_std).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
#         )
#         latents_ = noisy_latents * latents_std / pipeline.vae.config.scaling_factor + latents_mean
#     else:
#         latents_ = noisy_latents / pipeline.vae.config.scaling_factor

#     video = pipeline.vae.decode(latents_, return_dict=False)[0]
#     video = pipeline.video_processor.postprocess_video(video, output_type="pil")


# %%
# export_to_video(
#     video[0],
#     "output.mp4",
#     fps=30
# )

# %%
from diffusers.utils import export_to_video
video = pipeline(latents=noisy_latents.cuda(), 
         num_inference_steps=20,
         start_index=15,
         prompt=prompt, 
         negative_prompt="this is a nagative prompt",
         emphasize_indices=(0,0), 
         emphasize_neg_indices=(0,0),
) 

# %%
export_to_video(video["frames"][0], "output.mp4")

# %%
frames = torchvision.io.read_video(str(video_path), output_format="THWC", pts_unit="secs")[0]
extracted_positive_maps = [] 
extracted_negative_maps = []
maps = pipeline.attention_maps  
for step in range(max(len(maps) - 10, 0), len(maps)-1):
    for layer in range(len(maps[step])): 
        # print(maps[step][layer].shape)# [B, H, K, Q]
        map = maps[step][layer][0].mean(0)[0] # use sum, because in it it was softmax and spread outed
        # print(maps[step][layer][0].mean(0)[positive_mask==1].shape) # need to use bool!
        extracted_positive_maps.append(map.cpu().float().numpy().reshape(-1, frames[0].shape[0]//16, frames[0].shape[1]//16))
        
        map = maps[step][layer][0].mean(0)[1]
        extracted_negative_maps.append(map.cpu().float().numpy().reshape(-1, frames[0].shape[0]//16, frames[0].shape[1]//16))

        # assert torch.all(maps[step][layer][0].mean(0).sum(0) >= 0.99), maps[step][layer][0].mean(0).sum(0)
extracted_positive_maps = np.array(extracted_positive_maps)
extracted_negative_maps = np.array(extracted_negative_maps)
np.isnan(extracted_positive_maps).any()

# %%
import numpy as np
import math
file_id = 1
# %%
mask = (extracted_positive_maps > extracted_positive_maps.mean(axis=0) + extracted_positive_maps.std(axis=0)).astype(np.float32) + (extracted_positive_maps < extracted_positive_maps.mean(axis=0) - extracted_positive_maps.std(axis=0)).astype(np.float32)
mean_pos_map = np.nanmean(extracted_positive_maps, axis=0)


mask = (extracted_negative_maps > extracted_negative_maps.mean(axis=0) + extracted_negative_maps.std(axis=0)).astype(np.float32) + (extracted_negative_maps < extracted_negative_maps.mean(axis=0) - extracted_negative_maps.std(axis=0)).astype(np.float32)
mean_neg_map = np.nanmean(extracted_negative_maps, axis=0)

import cv2


def opening(x, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    x = [cv2.morphologyEx(i, cv2.MORPH_OPEN, kernel) for i in x]
    return x

def blur(x, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    x = [cv2.filter2D(i, -1, kernel) for i in x]
    return x

def resize(x, size):
    return [cv2.resize(i, size) for i in x]
    
    
def normalize(x):
    return [(i - i.min()) / (i.max() - i.min()) for i in x]

def compose_frames(frames, maps):
    frames = [np.array(frame).astype(float)/255 for frame in frames]
    for i in range(len(frames)):
        map = cv2.cvtColor(maps[(i-2)//6], cv2.COLOR_GRAY2RGB).astype(float)       
        frames[i] = cv2.addWeighted(frames[i], 0.5, map, 0.5, 0)
    return frames

def repeat_maps(maps, total_len):
    export_maps = []
    for i in range(total_len):
        idx = min(math.ceil((i-2)/6), len(maps)-1)
        export_maps.append(maps[idx])
    return export_maps

# print("a")
mean_pos_map = opening(mean_pos_map, kernel_size=3)
mean_neg_map = opening(mean_neg_map, kernel_size=3)
mean_pos_map = blur(mean_pos_map, kernel_size=3)
mean_neg_map = blur(mean_neg_map, kernel_size=3)
mean_pos_map = resize(mean_pos_map, size=(frames[0].shape[1], frames[0].shape[0]))
mean_neg_map = resize(mean_neg_map, size=(frames[0].shape[1], frames[0].shape[0]))

mean_pos_map = normalize(mean_pos_map)
mean_neg_map = normalize(mean_neg_map)

video = repeat_maps(mean_pos_map, len(frames))

# video = compose_frames(frames, mean_pos_map)
print(np.array(video).shape)
export_to_video(video, f"res/mochi_pos_{file_id:02d}_map.mp4", fps=30)

video = repeat_maps(mean_neg_map, len(frames))
# video = compose_frames(frames, mean_neg_map)
export_to_video(video, f"res/mochi_neg_{file_id:02d}_map.mp4", fps=30)

# fg = np.clip(np.array(mean_pos_map) - np.array(mean_neg_map), 0, 100)
fg = np.array(mean_pos_map) - np.array(mean_neg_map) * 2
fg[fg < 0] = 0
print(fg.max())

fg = normalize(fg)
video = compose_frames(frames, fg) 
export_to_video(video, f"map.mp4", fps=30)
wandb.log({"original": wandb.Video(video_path, fps=30), "fg": wandb.Video(f"map.mp4", fps=30)})
# %%
frames[0].shape, fg[0].shape


