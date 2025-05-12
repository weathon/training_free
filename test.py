# %%
import torch
from mochi_pipeline import MochiPipeline
from diffusers.utils import export_to_video

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
negative_end_index = tokenizer.tokenize(prompt).index("*")
print(tokenizer.tokenize(prompt))
print(start_index, end_index)

# %%
for block in pipe.transformer.transformer_blocks:
    block.attn1.processor = MochiAttnProcessor2_0(token_index_of_interest=torch.tensor(list(range(start_index + 1, end_index)))) #here start_index + 1, end_index, because exclude the *
    # block.attn1.processor = MochiAttnProcessor2_0(token_index_of_interest=torch.tensor([index])) 
frames = pipe(prompt,
              negative_prompt="A non-animal object being standing out, with its colour or texture contrast against the background such that it is highly visible with vibrant tones. The object is motionless.",
              num_inference_steps=30,
              guidance_scale=9,
              num_frames=60).frames[0]

export_to_video(frames, f"mochi_{file_id:02d}.mp4", fps=30)

# %%
frames[0].size[0]//16 * frames[0].size[1]//16

# %%
import pylab
import numpy as np

extracted_maps = []
maps = pipe.attention_maps  
for step in range(len(maps) - 10, len(maps)):
    for layer in range(len(maps[step])):
        # print(maps[step][layer].shape) [B, H, L, D]
        map = maps[step][layer][0].mean(0)[0]
        extracted_maps.append(map.cpu().float().numpy().reshape(-1, frames[0].size[1]//16, frames[0].size[0]//16))
extracted_maps = np.array(extracted_maps)

# %%
np.save(f"extracted_maps_{file_id:02d}.npy", extracted_maps)

# %%
mask = (extracted_maps > extracted_maps.mean(axis=0) + extracted_maps.std(axis=0)).astype(np.float32) + (extracted_maps < extracted_maps.mean(axis=0) - extracted_maps.std(axis=0)).astype(np.float32)
extracted_maps[mask != 0] = np.nan
mean_map = np.nanmean(np.abs(extracted_maps), axis=0)

# %%
mean_map.shape

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
        export_maps.append(maps[(i-2)//6])
    return export_maps
    
mean_map = opening(mean_map, kernel_size=3)
mean_map = blur(mean_map, kernel_size=3)
mean_map = resize(mean_map, size=(frames[0].size[0], frames[0].size[1]))

mean_map = normalize(mean_map)
video = repeat_maps(mean_map, len(frames))
export_to_video(video, f"mochi_{file_id:02d}_map.mp4", fps=30)


with open("file_id.txt", "w") as f:
    f.write(str(file_id + 1))
