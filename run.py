import os
import subprocess
import random
from shutil import copy
path = "prompts"
files = sorted(os.listdir(path))
run_id = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))
print(f"Run ID: {run_id}")
random.seed(42)
random.shuffle(files)
print(f"Found {len(files)} files")
os.makedirs("original_moca", exist_ok=True)

import argparse
parser = argparse.ArgumentParser(description="Mochi")
parser.add_argument("--negative_guidance_scale", type=float, default=3)
parser.add_argument("--triplet", action="store_true")
args = parser.parse_args()
negative_guidance_scale = args.negative_guidance_scale


for file in files[:10]:
    full_file_path = os.path.join(path, file)
    subprocess.run(["python", "mochi_main.py", "--filename", full_file_path, "--run_id", run_id, "--negative_guidance_scale", str(negative_guidance_scale), "--triplet" if args.triplet else ""])

    