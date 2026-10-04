import os
import sys
import subprocess
import shutil


REPO_URL = "https://github.com/MouaadErrougbani/brain-breast-tumor-segmentation-dl.git"

SOURCE_DATA = (
    "/kaggle/input/datasets/"
    "mouaaderg/breast-and-brain-tumor-segmentation/data"
)

WORK_DIR = "/kaggle/working"
REPO_NAME = "brain-breast-tumor-segmentation-dl"
REPO_DIR = os.path.join(WORK_DIR, REPO_NAME)

DEST_DATA = os.path.join(REPO_DIR, "data/splits")


# Clone
subprocess.run(
    ["git", "clone", REPO_URL, REPO_DIR],
    check=True
)


# Check repository
print("Repository:", REPO_DIR)
print("ML exists:", os.path.exists(os.path.join(REPO_DIR, "ml")))
print("Repository files:", os.listdir(REPO_DIR))


# Copy data
shutil.copytree(
    SOURCE_DATA,
    DEST_DATA,
    dirs_exist_ok=True
)


# Add repository to Python path
sys.path.insert(0, REPO_DIR)

os.chdir(REPO_DIR)


# Train
from ml.training.train import train

train()