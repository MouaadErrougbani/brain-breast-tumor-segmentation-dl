from pathlib import Path
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset


class TumorDataset() : 

    def __init__(self, images_dir, masks_dir):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        self.images = sorted(self.images_dir.glob("*.png"))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = Path(self.masks_dir, f'{image_path.stem}_mask{image_path.suffix}')

        image = self.to_tensor(Image.open(image_path).convert("L"))
        mask = self.to_tensor(Image.open(mask_path).convert("L"))

        mask = (mask > 0.5).float()
        return image, mask

        

    