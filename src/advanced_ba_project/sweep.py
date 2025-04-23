import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

import wandb
from advanced_ba_project.data import get_dataloaders
from advanced_ba_project.logger import log
from advanced_ba_project.model import UNet
from advanced_ba_project.train import DiceBCELoss, train_model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sweep_train():
    wandb.init()
    config = wandb.config

    set_seed(config.seed)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_path=Path("data/raw/Forest Segmented"),
        metadata_file="meta_data.csv",
        roboflow_train_path=Path("data/raw/roboflow/train"),
        roboflow_val_path=Path("data/raw/roboflow/valid"),
        roboflow_test_path=Path("data/raw/roboflow/test"),
        batch_size=config.batch_size,
        subset=False,
        seed=config.seed,
        apply_augmentation=config.apply_augmentation,
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        init_features=config.init_features,
        dropout_rate=config.dropout_rate,
    )

    criterion = DiceBCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.num_epochs,
        device=device,
    )

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    total_pixels = 0
    metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()

            preds = torch.sigmoid(outputs)
            preds_bin = (preds > 0.5).float()

            preds_flat = preds_bin.cpu().numpy().flatten().astype(int)
            masks_flat = masks.cpu().numpy().flatten().astype(int)

            batch_pixels = preds_flat.shape[0]
            total_pixels += batch_pixels

            metrics["accuracy"] += accuracy_score(masks_flat, preds_flat) * batch_pixels
            metrics["precision"] += precision_score(masks_flat, preds_flat, zero_division=0) * batch_pixels
            metrics["recall"] += recall_score(masks_flat, preds_flat, zero_division=0) * batch_pixels
            metrics["f1"] += f1_score(masks_flat, preds_flat, zero_division=0) * batch_pixels
            metrics["iou"] += jaccard_score(masks_flat, preds_flat, zero_division=0) * batch_pixels

    test_loss /= len(test_loader)
    for k in metrics:
        metrics[k] /= total_pixels

    wandb.log({
        "Test Loss": test_loss,
        "Test Accuracy": metrics["accuracy"],
        "Test Precision": metrics["precision"],
        "Test Recall": metrics["recall"],
        "Test F1": metrics["f1"],
        "Test IoU": metrics["iou"]
    })

    run_name = wandb.run.name.replace(" ", "_")
    model_path = f"models/unet_model_{run_name}.pth"
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)
    print(f"Model saved to {model_path}")

    wandb.finish()


if __name__ == "__main__":
    sweep_train()
