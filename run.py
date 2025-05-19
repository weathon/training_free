import os
import subprocess
import random
from shutil import copy
path = "prompts"
files = sorted(os.listdir(path))
print(f"Found {len(files)} files")
run_id = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))
print(f"Run ID: {run_id}")
os.makedirs("original_moca", exist_ok=True)

for file in files:
    full_file_path = os.path.join(path, file)
    subprocess.run(["python", "mochi_main.py", "--filename", full_file_path, "--run_id", run_id]) 
    # moca_image = os.path.join("/mnt/fastdata/original_moca/MoCA/JPEGImages", file, "00001.jpg")
    # copy(moca_image, os.path.join("original_moca", file + ".jpg"))
    