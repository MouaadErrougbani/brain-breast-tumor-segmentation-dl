from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TumorDataset(Dataset):
    def __init__(self, images_dir, masks_dir, apply_transform=True, normalize=True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.images = sorted(self.images_dir.glob("*.png"))

        if normalize:
            end_ops = [A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()]   # [-1, 1]
        else:
            end_ops = [A.ToFloat(max_value=255.0), ToTensorV2()]             # [0, 1]

        if apply_transform:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                                   rotate_limit=15, p=0.5),
                A.ElasticTransform(alpha=120, sigma=6, p=0.2),
                A.GridDistortion(p=0.2),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.GaussNoise(p=0.3),
                *end_ops,
            ])
        else:
            self.transform = A.Compose(end_ops)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks_dir / f"{image_path.stem}_mask{image_path.suffix}"

        image = np.array(Image.open(image_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)

        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].unsqueeze(0).float()