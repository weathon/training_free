# %%
import torch
from mochi_pipeline import MochiPipeline
from diffusers.utils import export_to_video
import math
pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.bfloat16).to("cuda")
# pipe.load_lora_weights("weathon/mochi-lora", adapter_name="camflagued")
# pipe.set_adapters(["camflagued"], adapter_weights=[0.5]) 
print("Loaded model")
import os
import wandb
from pdb import set_trace as bp

os.environ["TOKENIZERS_PARALLELISM"]="false"
# Enable memory savings
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# %%



from mochi_processor import MochiAttnProcessor2_0
from transformers import T5TokenizerFast
tokenizer = T5TokenizerFast.from_pretrained("genmo/mochi-1-preview", subfolder="tokenizer")

import argparse
import json
parser = argparse.ArgumentParser(description="Mochi")
parser.add_argument("--filename", type=str)
parser.add_argument("--run_id", type=str, default="mochi")
parser.add_argument("--negative_guidance_scale", type=float, default=3)
parser.add_argument("--triplet", action="store_true")
args = parser.parse_args()
filename = args.filename
moca_image = os.path.join("original_moca", filename.replace("prompts/","")+".jpg")
assert os.path.exists(moca_image), f"Image {moca_image} does not exist"

wandb.init(project="mochi", resume=True, id=args.run_id)

with open(filename, "r") as f:
    _prompt = f.read()
    
prompt = json.loads(_prompt)["pos_prompt"]
neg_prompt = json.loads(_prompt)["neg_prompt"]
print(prompt)
try:
    with open("file_id.txt", "r") as f:
        file_id = int(f.read())
except FileNotFoundError:
    file_id = 1
    
print(tokenizer.tokenize(prompt))

positive_start_index = tokenizer.tokenize(prompt).index("▁*")
positive_end_index = tokenizer.tokenize(prompt).index("*")

negative_start_index = tokenizer.tokenize(prompt).index("▁+")
negative_end_index = tokenizer.tokenize(prompt).index("+")

emphasize_start_index = tokenizer.tokenize(prompt).index("▁(")
try:
    emphasize_end_index = tokenizer.tokenize(prompt).index(")")
except:
    emphasize_end_index = tokenizer.tokenize(prompt).index(").")


negative_prompt = neg_prompt + " deformabled, (standing out, colour or texture contrast against the background, revealed, uncovered, exhibit, big and center, tha main object looks nothing like the background) blury, out of focus, pixelated, low resolution, low quality, low detail, low fidelity, low definition, high contrast, foggy, unnature, abnormal, rendered, odd, strange look, morph in, dirty lens, noisy"

emphasize_neg_start_index = tokenizer.tokenize(negative_prompt).index("▁(")
emphasize_neg_end_index = tokenizer.tokenize(negative_prompt).index(")")


indices = torch.tensor(list(range(positive_start_index + 1, positive_end_index)) + list(range(negative_start_index + 1, negative_end_index)))# + 1 # plus one because of the <extra_id_0> token
print(indices)
positive_mask = torch.tensor([1] * (positive_end_index - positive_start_index - 1) + [0] * (negative_end_index - negative_start_index - 1))
negative_mask = torch.tensor([0] * (positive_end_index - positive_start_index - 1) + [1] * (negative_end_index - negative_start_index - 1))
print(positive_mask)
print(negative_mask)
import random
# %%
pipe.indices = indices
pipe.positive_mask = positive_mask

# modify the scheduler?  
for block in pipe.transformer.transformer_blocks:
    block.attn1.processor = MochiAttnProcessor2_0(token_index_of_interest=indices, positive_mask=positive_mask) #here start_index + 1, end_index, because exclude the *
    # block.attn1.processor = MochiAttnProcessor2_0(token_index_of_interest=torch.tensor([index])) 

base = "high quality, 8k, nature, photo realistic, clear lens, in focus"

frames = pipe(prompt + base + ("" if random.random()<0.5 else " The animal is small and far away, making it even harder to see. "),
            negative_prompt=negative_prompt, 
            num_inference_steps=32,
            guidance_scale=6, 
            emphasize_indices=(emphasize_start_index, emphasize_end_index),
            emphasize_neg_indices=(emphasize_neg_start_index, emphasize_neg_end_index),
            generator=torch.manual_seed(19890604),
            negative_guidance_scale=args.negative_guidance_scale,
            text_weight=1,
            triplet=args.triplet,
            num_frames=37).frames[0]
# lower guidance scale, blury but camflagued
export_to_video(frames, f"res/mochi_{file_id:02d}.mp4", fps=30)
# blury image might be from the negative prompt 
# # export to frames
# import os
# os.makedirs(f"res/frames/{file_id:02d}", exist_ok=True)
# for i, frame in enumerate(frames):
#     frame.save(f"res/frames/{file_id:02d}/{i:04d}.png")
    


# %%
import pylab
import numpy as np


def softmax(x, axis=0):
    e_x = np.exp(x - np.max(x, axis=axis))
    return e_x / e_x.sum(axis=axis)


extracted_positive_maps = []
extracted_negative_maps = []
import tqdm
maps = pipe.attention_maps  
for step in tqdm.tqdm(range(max(len(maps) - 10, 0), len(maps))):
    for layer in range(len(maps[step])): 
        # print(maps[step][layer].shape)# [B, H, K, Q]
        map = maps[step][layer][0].mean(0)[0] # use sum, because in it it was softmax and spread outed
        # print(maps[step][layer][0].mean(0)[positive_mask==1].shape) # need to use bool!
        extracted_positive_maps.append(map.cpu().float().numpy().reshape(-1, frames[0].size[1]//16, frames[0].size[0]//16))
        
        map = maps[step][layer][0].mean(0)[1]
        extracted_negative_maps.append(map.cpu().float().numpy().reshape(-1, frames[0].size[1]//16, frames[0].size[0]//16))

        # assert torch.all(maps[step][layer][0].mean(0).sum(0) >= 0.99), maps[step][layer][0].mean(0).sum(0)
extracted_positive_maps = np.array(extracted_positive_maps)
extracted_negative_maps = np.array(extracted_negative_maps)

# print(extracted_positive_maps.shape)

# extracted_maps = np.stack([extracted_positive_maps, extracted_negative_maps], axis=0) / np.sqrt(128)
# extracted_maps = softmax(extracted_maps, axis=0)
# extracted_positive_maps = extracted_maps[0]
# extracted_negative_maps = extracted_maps[1]


negative_prompt_maps = torch.stack(pipe.negative_weights).cpu().float().numpy().mean(0).mean(0)



# %%
np.save(f"res/extracted_maps_pos_{file_id:02d}.npy", extracted_positive_maps)
np.save(f"res/extracted_maps_neg_{file_id:02d}.npy", extracted_negative_maps)
# print("extracted_positive_maps", extracted_positive_maps.shape)

# %%
mask = (extracted_positive_maps > extracted_positive_maps.mean(axis=0) + extracted_positive_maps.std(axis=0)).astype(np.float32) + (extracted_positive_maps < extracted_positive_maps.mean(axis=0) - extracted_positive_maps.std(axis=0)).astype(np.float32)
# extracted_positive_maps[mask != 0] = np.nan
mean_pos_map = extracted_positive_maps.mean(0)
# mean_pos_map = np.nanmean(extracted_positive_maps, axis=0)

mask = (extracted_negative_maps > extracted_negative_maps.mean(axis=0) + extracted_negative_maps.std(axis=0)).astype(np.float32) + (extracted_negative_maps < extracted_negative_maps.mean(axis=0) - extracted_negative_maps.std(axis=0)).astype(np.float32)
# extracted_negative_maps[mask != 0] = np.nan
mean_neg_map = extracted_negative_maps.mean(0)
# mean_neg_map = np.nanmean(extracted_negative_maps, axis=0)

# %%
# print("b")
# %%
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
        export_maps.append(maps[math.ceil((i-2)/6)])
    return export_maps

# print("a")
mean_pos_map = opening(mean_pos_map, kernel_size=3)
mean_neg_map = opening(mean_neg_map, kernel_size=3)
mean_pos_map = blur(mean_pos_map, kernel_size=3)
mean_neg_map = blur(mean_neg_map, kernel_size=3)
mean_pos_map = resize(mean_pos_map, size=(frames[0].size[0], frames[0].size[1]))
mean_neg_map = resize(mean_neg_map, size=(frames[0].size[0], frames[0].size[1]))

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
export_to_video(video, f"res/mochi_fg_{file_id:02d}_map.mp4", fps=30)
negative_prompt_maps = opening(negative_prompt_maps, kernel_size=3)
negative_prompt_maps = blur(negative_prompt_maps, kernel_size=3)
negative_prompt_maps = resize(negative_prompt_maps, size=(frames[0].size[0], frames[0].size[1]))
negative_prompt_maps = normalize(negative_prompt_maps)
negative_prompt_maps = repeat_maps(negative_prompt_maps, len(frames))
negative_prompt_maps = np.clip(negative_prompt_maps, 0, 1)

bp()
negative_prompt_maps = compose_frames(frames, negative_prompt_maps)
export_to_video(negative_prompt_maps, f"res/mochi_neg_prompt_{file_id:02d}_map.mp4", fps=30)

wandb.log({
    "video": wandb.Video(f"res/mochi_{file_id:02d}.mp4", caption=filename),
    "fg": wandb.Video(f"res/mochi_fg_{file_id:02d}_map.mp4"),
    "moca": wandb.Image(moca_image),
    "neg_prompt": wandb.Video(f"res/mochi_neg_prompt_{file_id:02d}_map.mp4"),
})
with open("file_id.txt", "w") as f:
    f.write(str(file_id + 1))
