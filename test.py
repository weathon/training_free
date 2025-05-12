# %%
import torch
from mochi_pipeline import MochiPipeline
from diffusers.utils import export_to_video
import math
pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", variant="bf16", torch_dtype=torch.bfloat16).to("cuda")

# Enable memory savings
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# %%

try:
    with open("file_id.txt", "r") as f:
        file_id = int(f.read())
except FileNotFoundError:
    file_id = 1






from processor import MochiAttnProcessor2_0
prompt = "A *devil scorpionfish* moves slowly across the seafloor, crawling with small, deliberate motions. The fish's body is excellently camouflaged, blending into the +rocky, algae-covered background+ with its mottled texture, muted colors, and irregular outline, making it very hard to see."
from transformers import T5TokenizerFast
tokenizer = T5TokenizerFast.from_pretrained("genmo/mochi-1-preview", subfolder="tokenizer")

# %%
positive_start_index = tokenizer.tokenize(prompt).index("▁*")
positive_end_index = tokenizer.tokenize(prompt).index("*")

negative_start_index = tokenizer.tokenize(prompt).index("▁+")
negative_end_index = tokenizer.tokenize(prompt).index("+")

print(tokenizer.tokenize(prompt))


indices = torch.tensor(list(range(positive_start_index + 1, positive_end_index)) + list(range(negative_start_index + 1, negative_end_index)))# + 1 # plus one because of the <extra_id_0> token
print(indices)
positive_mask = torch.tensor([[1] * (positive_end_index - positive_start_index - 1) + [0] * (negative_end_index - negative_start_index - 1)])
negative_mask = torch.tensor([[0] * (positive_end_index - positive_start_index - 1) + [1] * (negative_end_index - negative_start_index - 1)])
print(positive_mask)
print(negative_mask)

# %%
for block in pipe.transformer.transformer_blocks:
    block.attn1.processor = MochiAttnProcessor2_0(token_index_of_interest=indices) #here start_index + 1, end_index, because exclude the *
    # block.attn1.processor = MochiAttnProcessor2_0(token_index_of_interest=torch.tensor([index])) 
frames = pipe(prompt,
              negative_prompt="A non-animal object being standing out, with its colour or texture contrast against the background such that it is highly visible. The object is motionless or violent moving.",
              num_inference_steps=3,
              guidance_scale=9,
              num_frames=60).frames[0]

export_to_video(frames, f"res/mochi_{file_id:02d}.mp4", fps=30)

# %%
frames[0].size[0]//16 * frames[0].size[1]//16

# %%
import pylab
import numpy as np

extracted_positive_maps = []
extracted_negative_maps = []

maps = pipe.attention_maps  
for step in range(max(len(maps) - 10, 0), len(maps)):
    for layer in range(len(maps[step])):
        print(maps[step][layer].shape)# [B, H, Q, K]
        map = maps[step][layer][0].mean(0)[positive_mask].mean(0)
        extracted_positive_maps.append(map.cpu().float().numpy().reshape(-1, frames[0].size[1]//16, frames[0].size[0]//16))
        
        map = maps[step][layer][0].mean(0)[negative_mask].mean(0)
        extracted_negative_maps.append(map.cpu().float().numpy().reshape(-1, frames[0].size[1]//16, frames[0].size[0]//16))
        
extracted_positive_maps = np.array(extracted_positive_maps)
extracted_negative_maps = np.array(extracted_negative_maps)

# %%
np.save(f"res/extracted_maps_pos_{file_id:02d}.npy", extracted_positive_maps)
np.save(f"res/extracted_maps_neg_{file_id:02d}.npy", extracted_negative_maps)


# %%
mask = (extracted_positive_maps > extracted_positive_maps.mean(axis=0) + extracted_positive_maps.std(axis=0)).astype(np.float32) + (extracted_positive_maps < extracted_positive_maps.mean(axis=0) - extracted_positive_maps.std(axis=0)).astype(np.float32)
extracted_positive_maps[mask != 0] = np.nan
mean_pos_map = np.nanmean(np.abs(extracted_positive_maps), axis=0)

mask = (extracted_negative_maps > extracted_negative_maps.mean(axis=0) + extracted_negative_maps.std(axis=0)).astype(np.float32) + (extracted_negative_maps < extracted_negative_maps.mean(axis=0) - extracted_negative_maps.std(axis=0)).astype(np.float32)
extracted_negative_maps[mask != 0] = np.nan
mean_neg_map = np.nanmean(np.abs(extracted_negative_maps), axis=0)

# %%

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
    
mean_pos_map = opening(mean_pos_map, kernel_size=3)
mean_neg_map = opening(mean_neg_map, kernel_size=3)
mean_pos_map = blur(mean_pos_map, kernel_size=3)
mean_neg_map = blur(mean_neg_map, kernel_size=3)
mean_pos_map = resize(mean_pos_map, size=(frames[0].size[0], frames[0].size[1]))
mean_neg_map = resize(mean_neg_map, size=(frames[0].size[0], frames[0].size[1]))

mean_pos_map = normalize(mean_pos_map)
mean_neg_map = normalize(mean_neg_map)
video = repeat_maps(mean_pos_map, len(frames))
export_to_video(video, f"res/mochi_pos_{file_id:02d}_map.mp4", fps=30)
video = repeat_maps(mean_neg_map, len(frames))
export_to_video(video, f"res/mochi_neg_{file_id:02d}_map.mp4", fps=30)

with open("file_id.txt", "w") as f:
    f.write(str(file_id + 1))
