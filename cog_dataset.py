import os
import json
files = [i for i in os.listdir("dataset") if i.endswith(".txt")]

os.makedirs("cog_dataset", exist_ok=True)

with open("cog_dataset/videos.txt", "a") as f0:
    with open("cog_dataset/prompts.txt", "a") as f1:
        for file in files:
            with open(f"dataset/{file}", "r") as f2:
                prompt = json.load(f2)["pos_prompt"]
            f1.write(prompt + "\n")
            f0.write(file.replace(".txt", ".mp4") + "\n")
os.system("mkdir -p cog_dataset/videos")
os.system("mv dataset/*.mp4 cog_dataset/videos")
os.system("rm dataset/*.txt")
            