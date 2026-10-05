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


subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "segmentation-models-pytorch"
])

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

models = ["unet", "unet++", "deeplabv3"]
for model in models:
    train(model)

# Save results, then clean up
os.chdir(WORK_DIR)   # sortir du repo AVANT de le supprimer

dest_output = os.path.join(WORK_DIR, "output")
if os.path.exists(dest_output):
    shutil.rmtree(dest_output)
shutil.move(os.path.join(REPO_DIR, "output"), dest_output)

shutil.rmtree(REPO_DIR)   # supprime repo + data copiée

