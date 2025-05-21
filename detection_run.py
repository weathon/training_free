import os
import subprocess
import random
from shutil import copy
path = "test"
files = sorted(os.listdir(path))
run_id = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))
print(f"Run ID: {run_id}")
random.seed(42)
random.shuffle(files)
print(f"Found {len(files)} files")


for file in files[:10]:
    full_file_path = os.path.join(path, file)
    subprocess.run(["python", "detection.py", "--video_path", full_file_path, "--run_id", run_id])

    