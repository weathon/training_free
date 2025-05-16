import torch
from cog_pipeline import CogVideoXPipeline
from cog_processor import CogVideoXAttnProcessor2_0
from diffusers.utils import export_to_video
from transformers import T5TokenizerFast
import os 
import numpy as np
import math
import time

os.environ["TOKENIZERS_PARALLELISM"]="false"


prompt = "There is a *crab* blending into a +rocky ocean floor+ where the crab's mottled brown shell, rough texture, and uneven shape closely match the scattered rocks and coarse sand, all in muted brown and grey tones. The crab moves slowly and subtly, making it difficult to distinguish as its rough brown pattern looks just like a piece of rock among the uneven, similarly colored stones and patches of sand."
tokenizer = T5TokenizerFast.from_pretrained("THUDM/CogVideoX-5b", subfolder="tokenizer")

positive_start_index = tokenizer.tokenize(prompt).index("▁*")
positive_end_index = tokenizer.tokenize(prompt).index("*")

negative_start_index = tokenizer.tokenize(prompt).index("▁+")
negative_end_index = tokenizer.tokenize(prompt).index("+")

print(tokenizer.tokenize(prompt))
file_id = 1

indices = torch.tensor(list(range(positive_start_index + 1, positive_end_index)) + list(range(negative_start_index + 1, negative_end_index)))# + 1 # plus one because of the <extra_id_0> token
print(indices)
positive_mask = torch.tensor([1] * (positive_end_index - positive_start_index - 1) + [0] * (negative_end_index - negative_start_index - 1))
negative_mask = torch.tensor([0] * (positive_end_index - positive_start_index - 1) + [1] * (negative_end_index - negative_start_index - 1))
print(positive_mask)
print(negative_mask)

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)


pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

for block in pipe.transformer.transformer_blocks:
    block.attn1.processor = CogVideoXAttnProcessor2_0(token_index_of_interest=indices) 
    
frames = pipe(
    prompt=prompt,
    negative_prompt="standing out, colour or texture contrast against the background, visible, clear, distinct, easy to see, easy to distinguish, easy to identify, easy to recognize, easy to spot, easy to notice, easy to find, easy to detect, big and center",
    num_videos_per_prompt=1,
    num_inference_steps=30,
    num_frames=30,
    guidance_scale=12,
    generator=torch.Generator(device="cuda").manual_seed(int(time.time() * 1000) % 2**32),  
).frames[0]

export_to_video(frames, "output.mp4", fps=30)



extracted_positive_maps = []
extracted_negative_maps = []
import tqdm
maps = pipe.attention_maps  
for step in tqdm.tqdm(range(max(len(maps) - 10, 0), len(maps))):
    for layer in range(len(maps[step])): 
        # print(maps[step][layer].shape)# [B, H, K, Q]
        map = maps[step][layer][0].mean(0)[positive_mask==1].sum(0) # use sum, because in it it was softmax and spread outed
        # print(maps[step][layer][0].mean(0)[positive_mask==1].shape) # need to use bool!
        extracted_positive_maps.append(map.cpu().float().numpy().reshape(-1, frames[0].size[1]//16, frames[0].size[0]//16))
        
        map = maps[step][layer][0].mean(0)[negative_mask==1].sum(0) 
        extracted_negative_maps.append(map.cpu().float().numpy().reshape(-1, frames[0].size[1]//16, frames[0].size[0]//16))

        # assert torch.all(maps[step][layer][0].mean(0).sum(0) >= 0.99), maps[step][layer][0].mean(0).sum(0)
extracted_positive_maps = np.array(extracted_positive_maps)
extracted_negative_maps = np.array(extracted_negative_maps)

# print(extracted_positive_maps.shape)

# extracted_maps = np.stack([extracted_positive_maps, extracted_negative_maps], axis=0) / np.sqrt(128)
# extracted_maps = softmax(extracted_maps, axis=0)
# extracted_positive_maps = extracted_maps[0]
# extracted_negative_maps = extracted_maps[1]



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
export_to_video(video, f"res/cog_pos_{file_id:02d}_map.mp4", fps=30)

video = repeat_maps(mean_neg_map, len(frames))
# video = compose_frames(frames, mean_neg_map)
export_to_video(video, f"res/cog_neg_{file_id:02d}_map.mp4", fps=30)

# fg = np.clip(np.array(mean_pos_map) - np.array(mean_neg_map), 0, 100)
fg = np.array(mean_pos_map) - np.array(mean_neg_map) * 2
fg[fg < 0] = 0
print(fg.max())

fg = normalize(fg)
video = compose_frames(frames, fg)
export_to_video(video, f"res/cog_fg_{file_id:02d}_map.mp4", fps=30)

