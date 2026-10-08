import torch
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
import os

BASE_OUTPUT_FIRST = "./output"
BASE_OUTPUT_IMPROVED = "./output/improved"
models = ["unet", "unet++", "deeplabv3"]
metrics = ["train_losses", "val_losses", "train_dices", "val_dices",
           "train_ious", "val_ious", "times"]


def load_metrics(path):
    History = {}
    for model in models: 
        History[model] = {}
        for metric in metrics: 
            v = torch.load(Path(path, model, f'{metric}.pt'))
            History[model][metric] = np.array(v)
    return History


def validation_metric_curves(History, path):
    fig, ax = plt.subplots(1, 3, figsize=(18,5))
    for m in models:
        ax[0].plot(History[m]["val_losses"], label=m)
        ax[1].plot(History[m]["val_dices"], label=m)
        ax[2].plot(History[m]["val_ious"], label=m)

    for a, t in zip(ax, ["Val Loss", "Val Dice", "Val IoU"]):
        a.set_title(t); a.set_xlabel("Epoch")
        a.legend()
        
    plt.savefig(f'{path}/validation_metric.png')
    plt.close()

def training_metric_curves(History, path): 
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for m in models: 
        ax[0].plot(History[m]["train_losses"], label=m)
        ax[1].plot(History[m]["train_dices"], label=m)
        ax[2].plot(History[m]["train_ious"], label=m)

    for a, t in zip (ax, ["Train Loss", "Train Dice", "Train IoU"]):
        a.set_title(t)
        a.set_xlabel("Epoch")
        a.legend()

    plt.savefig(f'{path}/training_metric.png')
    plt.close()

def training_validation_courves(History, path):
    fig, ax = plt.subplots(3, 3, figsize=(18, 12), sharex=True)

    for idx, m in enumerate(models):
        ax[idx][0].plot(History[m]["train_losses"], label="Train Loss")
        ax[idx][0].plot(History[m]["val_losses"], label="Val Loss")
        ax[idx][1].plot(History[m]["train_dices"], label="Train Dice")
        ax[idx][1].plot(History[m]["val_dices"], label="Val Dice")
        ax[idx][2].plot(History[m]["train_ious"], label="Train IoU")
        ax[idx][2].plot(History[m]["val_ious"], label="Val IoU")

        ax[idx][0].set_ylabel(m, fontsize=13, fontweight="bold")

    for j, name in enumerate(["Loss", "Dice", "IoU"]):
        ax[0][j].set_title(name, fontsize=13)

    for axs in ax:
        for a in axs:
            a.legend()
            a.grid(alpha=0.3)

    for a in ax[-1]:
        a.set_xlabel("Epoch")

    fig.suptitle("Train vs Validation curves per model", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{path}/all_models_train_vs_val.png", dpi=300)
    plt.close()

def info_numerics(History):
    rows = []
    for m in models:
        h = History[m]
        b = h["val_dices"].argmax()               
        rows.append({
            "model": m,
            "best_epoch": b + 1,
            "val_dice": h["val_dices"][b],
            "val_iou": h["val_ious"][b],
            "val_loss": h["val_losses"][b],
            "train_dice": h["train_dices"][b],
            "gap_dice": h["train_dices"][b] - h["val_dices"][b],
            "time_per_epoch": h["times"][:,0].mean(),
            "total_time": h["times"][-1,-1],
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df)
        
def analysis():
    for path in [BASE_OUTPUT_FIRST, BASE_OUTPUT_IMPROVED]: 

        History = load_metrics(path)
        path_plots = Path(path, "plots")
        os.makedirs(path_plots, exist_ok=True)
        validation_metric_curves(History=History, path=path_plots)
        training_metric_curves(History=History, path=path_plots)
        training_validation_courves(Historys=History, path=path_plots)
        info_numerics(History=History)


if __name__ == "__main__":
    analysis()