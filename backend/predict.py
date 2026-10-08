import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

SIZE = (512, 512)
TRANSFORM = A.Compose([A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])


def build_model(name, n_class=1):
    kw = dict(encoder_name="resnet34", encoder_weights=None,
              in_channels=1, classes=n_class, activation=None)
    return {"unet": smp.Unet,
            "unet++": smp.UnetPlusPlus,
            "deeplabv3": smp.DeepLabV3}[name](**kw)


def preprocessing(img):
    img = Image.fromarray(img).convert("L")
    img = img.resize(SIZE, Image.Resampling.BICUBIC)   
    out = TRANSFORM(image=np.array(img))["image"]      
    return out.unsqueeze(0)    



@torch.inference_mode()
def predict(model, x, device="cpu", threshold=0.5):
    model.eval()
    logits = model(x)                                  # [1,1,H,W]
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()   # [H,W]
    return (prob > threshold).astype(np.uint8), prob
