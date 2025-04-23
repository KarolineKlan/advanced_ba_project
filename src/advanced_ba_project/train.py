import os
import random
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

import wandb
from advanced_ba_project.data import get_dataloaders
from advanced_ba_project.logger import log
from advanced_ba_project.model import UNet


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)

        smooth = 1e-6
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return bce_loss + dice_loss

# Define the training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=10, device="cuda"):
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        log.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        log.info(f"Train Loss: {epoch_loss:.4f}")
        wandb.log({"Train Loss": epoch_loss, "Epoch": epoch + 1})

        # Validation phase
        model.eval()
        val_loss = 0.0
        total_pixels = 0
        metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

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

        val_loss /= len(val_loader)
        for k in metrics:
            metrics[k] /= total_pixels

        val_losses.append(val_loss)
        log.info(f"Validation Loss: {val_loss:.4f}")
        log.info(
            f"Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | IoU: {metrics['iou']:.4f}"
        )

        wandb.log(
            {
                "Validation Loss": val_loss,
                "Val Accuracy": metrics["accuracy"],
                "Val Precision": metrics["precision"],
                "Val Recall": metrics["recall"],
                "Val F1": metrics["f1"],
                "Val IoU": metrics["iou"],
            },
            step=epoch + 1
        )

        if scheduler:
            scheduler.step(val_loss)

    return train_losses, val_losses


@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    log.info(f"Using Hydra Config: {cfg}")
    timestamp = cfg.timestamp
    set_seed(cfg.seed)
    wandb_config = OmegaConf.to_container(cfg, resolve=True)

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=f"{cfg.experiment_name}_{timestamp}",
        config=wandb_config,
        mode=cfg.wandb.mode,
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        data_path=Path(to_absolute_path(cfg.dataset.data_path)),
        metadata_file=cfg.dataset.metadata_file,
        roboflow_train_path=Path(to_absolute_path(cfg.dataset.roboflow_train_path)),
        roboflow_val_path=Path(to_absolute_path(cfg.dataset.roboflow_val_path)),
        roboflow_test_path=Path(to_absolute_path(cfg.dataset.roboflow_test_path)),
        batch_size=cfg.hyperparameters.batch_size,
        subset=cfg.dataset.subset,
        seed=cfg.seed,
        apply_augmentation=cfg.dataset.apply_augmentation,
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        init_features=cfg.model.init_features,
        dropout_rate=cfg.model.dropout_rate,
    )

    criterion = DiceBCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.hyperparameters.learning_rate,
        weight_decay=cfg.hyperparameters.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler=scheduler, num_epochs=cfg.hyperparameters.num_epochs, device=device
    )

    model_path = f"models/unet_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    log.success(f"Model saved as {model_path}")
    wandb.save(model_path)

    loss_plot_path = f"reports/figures/loss_plot_{timestamp}.png"
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig(loss_plot_path)
    log.success(f"Loss plot saved as {loss_plot_path}")
    wandb.log({"Loss Plot": wandb.Image(loss_plot_path)})

    # Final evaluation on the test set
    model.eval()
    test_loss = 0.0
    total_pixels = 0
    metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}

    # Evaluate on the test set
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

    # Calculate average metrics
    test_loss /= len(test_loader)
    for k in metrics:
        metrics[k] /= total_pixels

    # Log test metrics
    log.info(f"Test Loss: {test_loss:.4f}")
    log.info(
        f"Test Accuracy: {metrics['accuracy']:.4f} | Test Precision: {metrics['precision']:.4f} | "
        f"Test Recall: {metrics['recall']:.4f} | Test F1: {metrics['f1']:.4f} | Test IoU: {metrics['iou']:.4f}"
    )
    wandb.log(
        {
            "Test Loss": test_loss,
            "Test Accuracy": metrics["accuracy"],
            "Test Precision": metrics["precision"],
            "Test Recall": metrics["recall"],
            "Test F1": metrics["f1"],
            "Test IoU": metrics["iou"],
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()

