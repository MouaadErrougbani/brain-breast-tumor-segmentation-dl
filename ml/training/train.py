import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from time import time
import segmentation_models_pytorch as smp

from ml.utils.utils import load_config
from ml.data.tumor_dataset import TumorDataset
from ml.training.metrics import dice_score, iou_score

def train(model_name = None):

    # ==================== Configuration ====================
    config = load_config("./ml/config/train.yaml")
    seed = config.get("seed", 42)
    model_name = model_name if model_name is not None else config.get("model", "UNet")
    batch_size = config.get("batch_size", 8)
    epochs = config.get("epochs", 10)
    optimizer_name = config.get("optimizer", "adam")
    lr = config.get("learning_rate", 0.0001)
    device_mode = config.get("device", "auto")
    save_best = config.get("save_best", True)
    output_dir = config.get("output_dir", "./output")
    data_dir = config.get("data_dir", "./data/splits")
    n_class = config.get("n_class", 1)
    loss_function_name = config.get("loss_function", "BCE")
    num_workers = config.get("num_workers", 0)

    # Creat output directory     
    output_dir = Path(output_dir,model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add Seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ==================== Device ====================
    devices = {
        "cpu": torch.device("cpu"),
        "cuda": torch.device("cuda"),
        "auto": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    device = devices[device_mode.lower()]

    # ==================== Dataset ====================
    path_train_images = Path(data_dir, "train", "images")
    path_train_masks = Path(data_dir, "train", "masks")
    path_val_images = Path(data_dir, "val", "images")
    path_val_masks = Path(data_dir, "val", "masks")

    train_data = TumorDataset(path_train_images, path_train_masks)
    val_data = TumorDataset(path_val_images, path_val_masks)

    # ==================== DataLoader ====================
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    

    # ==================== Model ====================
    
    if model_name.lower() == "unet":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=n_class,
            activation=None
        )

    elif model_name.lower() == "unet++":
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=n_class,
            activation=None
        )

    elif model_name.lower() == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=n_class,
            activation=None
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # ==================== Loss ====================
    loss_functions = {
        "bce": BCEWithLogitsLoss()
    }
    criterion = loss_functions[loss_function_name.lower()]

    # ==================== Optimizer ====================
    optimizers = {
        "adam": Adam(model.parameters(), lr=lr)
    }
    optimizer = optimizers[optimizer_name.lower()]

    # ==================== Infos =========================
    print("=="*40)
    print()
    print(f"Device: {device}")
    if device.type == "cuda": 
        print(f"Using {torch.cuda.device_count()} GPUs")
    print(f"Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f'Number workers: {num_workers}')
    print()
    print("=="*40)
    print()

    # ==================== Training ====================
    best_dice = 0.0
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    train_ious = []
    val_ious = []
    times = []
    time_total = time()
    # Training loop
    for epoch in range(epochs):
        # ==================== TRAIN ====================
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        train_start = time()

        for images, masks in train_loader: 
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Forward
            outputs = model(images)

            # Loss 
            loss = criterion(outputs, masks)

            # Backward
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()
            train_dice += dice_score(outputs=outputs, masks=masks)
            train_iou  += iou_score(outputs=outputs, masks=masks)
        end_time = time()
        train_time = end_time - train_start
        all_time = end_time - time_total
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou  /= len(train_loader)

        # ==================== VALIDATION ====================

        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
       
        model.eval()

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                # Forward
                outputs = model(images)
                
                # Loss 
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_score(outputs, masks)
                val_iou += iou_score(outputs, masks)

               

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        

        print(
            f"[{model_name}] "
            f"Epoch {epoch + 1}/{epochs} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Dice: {val_dice:.4f} "
            f"| IoU: {val_iou:.4f} "
            f"| Time Epoch: {train_time}s "
            f"| All time: {all_time}s"

        )
        # ==================== Save Best ====================

        if save_best and (val_dice > best_dice) : 
            best_dice = val_dice
            torch.save(
                model.module.state_dict() if isinstance(model, torch.nn.DataParallel)
                else model.state_dict(),
                Path(output_dir, "best_model.pth")
            )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        times.append((train_time, all_time))

    # ==================== Save logs ====================
    torch.save(
        train_losses,
        Path(output_dir, "train_losses.pt")
    )
    torch.save(
        val_losses,
        Path(output_dir, "val_losses.pt")
    )

    torch.save(
        train_dices,
        Path(output_dir, "train_dices.pt")
    )
    torch.save(
        val_dices,
        Path(output_dir, "val_dices.pt")
    )
    
    torch.save(
        train_ious,
        Path(output_dir, "train_ious.pt")
    )
    torch.save(
        val_ious,
        Path(output_dir, "val_ious.pt")
    )


if __name__ == "__main__" : 
    models = ["unet", "unet++", "deeplabv3"]
    for model in models :
        train(model)