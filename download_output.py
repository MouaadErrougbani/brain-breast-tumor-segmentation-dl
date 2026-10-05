import kagglehub


models = ["deeplabv3", "unet", "unet++"]
files = ["best_model.pth", "times.pt", "train_dices.pt", "train_ious.pt", "train_losses.pt", "val_dices.pt", "val_ious.pt", "val_losses.pt"]
base_path = "brain-breast-tumor-segmentation-dl/output/"
output_dir="./output"

for model in models: 
    for file in files:            
        kagglehub.notebook_output_download(
            "mouaaderg/brain-breast-tumor-segmentation-unet",
            path=f"brain-breast-tumor-segmentation-dl/output/{model}/{file}",
            output_dir=f"{output_dir}/{model}"
        )