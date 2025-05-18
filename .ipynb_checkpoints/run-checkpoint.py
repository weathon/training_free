import os
import subprocess
import random
path = "../finetrainers/training/mochi-1/dataset/"
files = sorted(os.listdir(path))
files = [f for f in files if f.split("_")[-1] == "001.txt"]
print(f"Found {len(files)} files")
run_id = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))
print(f"Run ID: {run_id}")
for file in files:
    full_file_path = os.path.join(path, file)
    subprocess.run(["python", "mochi_main.py", "--filename", full_file_path, "--run_id", run_id])