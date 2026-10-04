import os
import shutil
from pathlib import Path

import kagglehub



def download_data():

    sources = ["tinashri/brain-tumor-dataset-includes-the-mask-and-images", "aryashah2k/breast-ultrasound-images-dataset"]
    output_dir = Path("data/raw")

    if not sources:
        print("No source data paths found.")
        return None


    output_dir.mkdir(parents=True, exist_ok=True)

    for source in sources:
        print(f"Downloading: {source}")

        downloaded_path = kagglehub.dataset_download(source)

        print(f"Downloaded to: {downloaded_path}")

        source_path = Path(downloaded_path)

        destination = output_dir / source.split("/")[-1]

        if destination.exists():
            print(f"Already exists: {destination}")
            continue

        shutil.move(str(source_path), str(destination))

        print(f"Moved to: {destination}")


if __name__ == "__main__":
    download_data()