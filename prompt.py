

from openai import OpenAI
from PIL import Image
import io
import os
import base64
from pydantic import BaseModel
from typing import List, Optional

class Prompt(BaseModel):
    pos_prompt: str
    neg_prompt: str
  
os.makedirs("./prompts", exist_ok=True)
positive_gpt_prompt = "Describe in 2-3 setence this image such that the prompt will be used to generate an video. Mention motion, the object might be hard to see. the filename is %s, filename might has other things than the animal. Do not mention filename just the animal name. The object might be hard to see (mention this) and blended into the envirement. Mention the animal is camflague and blend into the envirement for texture, color, and shape (mention the specific color texture and shape of both the animal and background separately, You should prioritize making the animal camflagued (mention color, texture, etc) befor realism and follow the real image, i.e. if the animal is gray and the background in the image is yellow, describe them both as gray or yellow. Make sure their color as close as possible. i.e. mention the animal has color A and the background has a close color A). Mention the animal's motion, but do not mention this makes it easy to spot. At the same time, provide a negative prompt, tell what the model should not generate, could be: clearly visible, standing out, easy to spot, obvious, distinct, high contrast, sharp outline, border, vibrant colors, unnatural colors, unnature bodies, blurry, low quality, pixelated, text, over exposure, etc. Do just just copy paste this, write it based on specific video. Both prompt should be description and not action requiring. Do not mention positive part in negative prompt and vice versa. Negative prompt should be description of what not to be in a positive way, e.g. 'ugly face', instead of 'do not generate ugly face'. Negative prompt should be information dense, not detialed, and just a list of words. Use simple language! Mention the foreground and the background as what they are, what the animal is blending into. When give your positive prompt, put the foreground object (animal) in *animal* and background key words in +background+, you can only have one foreground and one background highlighted, like 'there is *rabbit* ...... in a +bush+.  Mention what the animal looks like in the background, like the crab looks like a rock, blend into other rocks and sand. Mention this and the background sperately. Use positive phases like 'it is blended in, indistinct' etc not negative phases like 'it is not easy to see', 'hard to see', etc. "


    
def caption_image(imgs, filename):
    client = OpenAI()
    base64s = []
    for img in imgs:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"
        base64s.append(image_url)
        
    response = client.beta.chat.completions.parse(
    model="gpt-4.1",
    messages=[
        {
        "role": "user",
        "content": [
            *[{
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
            } for image_url in base64s],
            {
            "type": "text",
            "text": positive_gpt_prompt % filename
            }
        ]
        }, 
    ],
    response_format=Prompt,
    )
    return response.choices[0].message.content

import os
import cv2
videos = os.listdir("/mnt/fastdata/original_moca/MoCA/JPEGImages")

def get_prompt(video):
    video_path = os.path.join("/mnt/fastdata/original_moca/MoCA/JPEGImages", video)
    image_path = os.path.join("/mnt/fastdata/original_moca/MoCA/JPEGImages", video_path, "00001.jpg")
    imgs = [Image.open(image_path)]
    # Capture image and get prompt
    prompt = caption_image(imgs, video) 
    
    with open(os.path.join("./prompts", video), "w") as f:
        f.write(prompt)
    print(prompt)
    return prompt
# videos = [i for i in os.listdir("./prompts") if not "seg" in i]
print(len(videos)) 
# use a pool of 30 processes
from multiprocessing import Pool
if __name__ == "__main__":
    with Pool(30) as p:
        prompts = p.map(get_prompt, videos)
    print(prompts)

# get_prompt("crab_1_000.mp4")