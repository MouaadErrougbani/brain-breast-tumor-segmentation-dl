import os
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


# ==================== Clone repository ====================

subprocess.run(
    ["git", "clone", REPO_URL, REPO_DIR],
    check=True
)


# ==================== Copy dataset ====================

shutil.copytree(
    SOURCE_DATA,
    DEST_DATA,
    dirs_exist_ok=True
)


# ==================== Run training ====================

os.chdir(REPO_DIR)

from ml.training.train import train

train()