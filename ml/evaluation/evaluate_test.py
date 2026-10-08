import time
import numpy as np
import torch
import segmentation_models_pytorch as smp


def build_model(name, n_class=1):
    kw = dict(encoder_name="resnet34", encoder_weights=None,
              in_channels=1, classes=n_class, activation=None)
    return {"unet": smp.Unet,
            "unet++": smp.UnetPlusPlus,
            "deeplabv3": smp.DeepLabV3}[name](**kw)


def per_image_scores(outputs, masks, threshold=0.5, smooth=1e-7):
    # Same logic as dice_score / iou_score, without the mean
    preds = (torch.sigmoid(outputs) > threshold).float()
    preds = preds.view(preds.size(0), -1)
    masks = masks.view(masks.size(0), -1)
    inter = (preds * masks).sum(1)
    dice = (2 * inter + smooth) / (preds.sum(1) + masks.sum(1) + smooth)
    iou = (inter + smooth) / (preds.sum(1) + masks.sum(1) - inter + smooth)
    return dice.cpu().numpy(), iou.cpu().numpy()


@torch.no_grad()
def evaluate(model_name, weights_path, loader, device):
    model = build_model(model_name).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    dices, ious = [], []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        d, i = per_image_scores(model(images), masks)
        dices.append(d); ious.append(i)

    # Inference time: batch size 1, with warm-up and GPU sync
    x = next(iter(loader))[0][:1].to(device)
    for _ in range(10):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / 50 * 1000

    return {
        "dice": np.concatenate(dices),
        "iou": np.concatenate(ious),
        "params": sum(p.numel() for p in model.parameters()),
        "inference_ms": ms,
    }