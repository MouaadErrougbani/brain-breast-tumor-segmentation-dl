import torch


def dice_score(outputs, masks, threshold=0.5, smooth=1e-7):
    """
    Compute Dice Score for binary segmentation.
    """

    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    masks = masks.view(masks.size(0), -1)

    intersection = (preds * masks).sum(dim=1)

    dice = (
        (2 * intersection + smooth)
        / (preds.sum(dim=1) + masks.sum(dim=1) + smooth)
    )

    return dice.mean().item()


def iou_score(outputs, masks, threshold=0.5, smooth=1e-7):
    """
    Compute IoU (Intersection over Union) for binary segmentation.
    """

    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    masks = masks.view(masks.size(0), -1)

    intersection = (preds * masks).sum(dim=1)

    union = (
        preds.sum(dim=1)
        + masks.sum(dim=1)
        - intersection
    )

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()