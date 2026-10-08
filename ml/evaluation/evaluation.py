import torch
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd


from ml.data.tumor_dataset import TumorDataset
from ml.evaluation.evaluate_test import evaluate


def evalutation():
    
        path_test_images = "./data/splits/test/images" 
        path_test_masks = "./data/splits/test/masks" 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # One loader per preprocessing setup (see the check above)
        test_loader_new = DataLoader(TumorDataset(path_test_images, path_test_masks, apply_transform=False),
                                batch_size=16, shuffle=False)
        test_loader_old = DataLoader(TumorDataset(path_test_images, path_test_masks, apply_transform=False, normalize=False),
                                batch_size=16, shuffle=False)

        results = {}
        for setup, root, loader in [("baseline", "output", test_loader_old),
                                ("improved", "output/improved", test_loader_new)]:
                for m in ["unet", "unet++", "deeplabv3"]:
                        results[(m, setup)] = evaluate(m, Path(root)/m/"best_model.pth", loader, device)

        rows = [{"model": m, "setup": s,
                "test_dice": r["dice"].mean(), "dice_std": r["dice"].std(),
                "test_iou": r["iou"].mean(), "iou_std": r["iou"].std(),
                "params_M": r["params"]/1e6, "inference_ms": r["inference_ms"]}
                for (m, s), r in results.items()]
        print(pd.DataFrame(rows).round(4))

if __name__ == "__main__":
        evalutation()