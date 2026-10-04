import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import tqdm
import segmentation_models_pytorch as smp

from ml.utils.utils import load_config
from ml.data.tumor_dataset import TumorDataset
from ml.training.metrics import dice_score, iou_score

def train():

    # ==================== Configuration ====================
    config = load_config("./ml/config/train.yaml")
    seed = config.get("seed", 42)
    model_name = config.get("model", "UNet")
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

    # Creat output directory     
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Add Seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ==================== Dataset ====================
    path_train_images = Path(data_dir, "train", "images")
    path_train_masks = Path(data_dir, "train", "masks")
    path_val_images = Path(data_dir, "val", "images")
    path_val_masks = Path(data_dir, "val", "masks")

    train_data = TumorDataset(path_train_images, path_train_masks)
    val_data = TumorDataset(path_val_images, path_val_masks)

    # ==================== DataLoader ====================
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ==================== Device ====================
    devices = {
        "cpu": torch.device("cpu"),
        "cuda": torch.device("cuda"),
        "auto": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    device = devices[device_mode.lower()]

    # ==================== Model ====================
    
    models = {

        "unet": smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=n_class,
            activation=None
        ),

        "unet++": smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=n_class,
            activation=None
        ),

        "deeplabv3": smp.DeepLabV3(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=n_class,
            activation=None
        )
    }
    model = models[model_name.lower()].to(device)

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
    
    # ==================== Training ====================
    best_dice = 0.0
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    train_ious = []
    val_ious = []

    # Training loop
    for epoch in range(epochs):
        # ==================== TRAIN ====================
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for images, masks in progress_bar: 
            images = images.to(device)
            masks = masks.to(device)
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

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou  /= len(train_loader)

        # ==================== VALIDATION ====================

        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0

        progress_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )
        model.eval()

        with torch.no_grad():
            for images, masks in progress_bar:
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward
                outputs = model(images)
                
                # Loss 
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_score(outputs, masks)
                val_iou += iou_score(outputs, masks)

                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Dice: {val_dice:.4f} "
            f"| IoU: {val_iou:.4f}"
        )
        # ==================== Save Best ====================

        if save_best and (val_dice > best_dice) : 
            best_dice = val_dice
            torch.save(
                model.state_dict(),
                Path(output_dir, "best_model.pth")
            )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

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
    train()