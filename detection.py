# import os

# from diffusers import AutoencoderKLMochi
# from mochi_pipeline import MochiPipeline
# from transformers import T5EncoderModel, T5Tokenizer
# from mochi_processor import MochiAttnProcessor2_0
# import torch
# import torchvision
# from pathlib import Path
# from diffusers.utils import export_to_video
# os.environ["TOKENIZERS_PARALLELISM"]="false"

# model_id = "genmo/mochi-1-preview"
# pipeline = MochiPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# pipeline.enable_vae_slicing()
# pipeline.enable_vae_tiling()
# pipeline.enable_model_cpu_offload()

# import numpy as np
# from transformers import T5TokenizerFast
# tokenizer = T5TokenizerFast.from_pretrained("genmo/mochi-1-preview", subfolder="tokenizer")
# prompt = "a (camflagued) *animal* moving in the +background+" #fuck, the prompt problem? cannot have end? saw index and then 
# positive_start_index = tokenizer.tokenize(prompt).index("▁*")
# positive_end_index = tokenizer.tokenize(prompt).index("*")

# negative_start_index = tokenizer.tokenize(prompt).index("▁+")
# negative_end_index = tokenizer.tokenize(prompt).index("+")

# print(tokenizer.tokenize(prompt))


# indices = torch.tensor(list(range(positive_start_index + 1, positive_end_index)) + list(range(negative_start_index + 1, negative_end_index)))# + 1 # plus one because of the <extra_id_0> token
# print(indices)
# positive_mask = torch.tensor([1] * (positive_end_index - positive_start_index - 1) + [0] * (negative_end_index - negative_start_index - 1))
# negative_mask = torch.tensor([0] * (positive_end_index - positive_start_index - 1) + [1] * (negative_end_index - negative_start_index - 1))
# print(positive_mask)
# print(negative_mask)
# pipeline.indices = indices
# pipeline.positive_mask = positive_mask

# emphasize_start_index = tokenizer.tokenize(prompt).index("▁(")
# try:
#     emphasize_end_index = tokenizer.tokenize(prompt).index(")")
# except:
#     emphasize_end_index = tokenizer.tokenize(prompt).index(").")


# negative_prompt = "deformabled, (standing out, colour or texture contrast against the background, revealed, uncovered, exhibit, big and center, tha main object looks nothing like the background) blury, out of focus, pixelated, low resolution, low quality, low detail, low fidelity, low definition, high contrast, foggy, unnature, abnormal, rendered, odd, strange look, morph in, dirty lens, noisy"

# emphasize_neg_start_index = tokenizer.tokenize(negative_prompt).index("▁(")
# emphasize_neg_end_index = tokenizer.tokenize(negative_prompt).index(")")




# for block in pipeline.transformer.transformer_blocks:
#     block.attn1.processor = MochiAttnProcessor2_0(token_index_of_interest=indices, positive_mask=positive_mask) 

# video = pipeline(prompt,
#             negative_prompt=negative_prompt, 
#             num_inference_steps=10,
#             guidance_scale=6,
#             emphasize_indices=(emphasize_start_index, emphasize_end_index),
#             emphasize_neg_indices=(emphasize_neg_start_index, emphasize_neg_end_index),
#             generator=torch.manual_seed(19890604),
#             negative_guidance_scale=6,
#             num_frames=37)

# export_to_video(video.frames[0], "a.mp4", fps=30)

import torch
from mochi_pipeline import MochiPipeline
from diffusers.utils import export_to_video
import math
pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.bfloat16).to("cuda")
print("Loaded model")
import os
import wandb

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

    
prompt = "a (camflagued) *animal* moving in the +background+"
neg_prompt = "deformabled, (standing out, colour or texture contrast against the background, revealed, uncovered, exhibit, big and center, tha main object looks nothing like the background) blury, out of focus, pixelated, low resolution, low quality, low detail, low fidelity, low definition, high contrast, foggy, unnature, abnormal, rendered, odd, strange look, morph in, dirty lens, noisy"

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

negative_prompt = neg_prompt
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

for block in pipe.transformer.transformer_blocks:
    block.attn1.processor = MochiAttnProcessor2_0(token_index_of_interest=indices, positive_mask=positive_mask) #here start_index + 1, end_index, because exclude the *

base = "high quality, 8k, nature, photo realistic, clear lens, in focus"

frames = pipe(prompt + base + ("" if random.random()<0.5 else " The animal is small and far away, making it even harder to see. "),
            negative_prompt=negative_prompt, 
            num_inference_steps=64,
            guidance_scale=6,
            emphasize_indices=(emphasize_start_index, emphasize_end_index),
            emphasize_neg_indices=(emphasize_neg_start_index, emphasize_neg_end_index),
            generator=torch.manual_seed(19890604),
            negative_guidance_scale=6,
            num_frames=37).frames[0]
# lower guidance scale, blury but camflagued
export_to_video(frames, f"a.mp4", fps=30)