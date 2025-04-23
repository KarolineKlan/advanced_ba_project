import argparse
import datetime
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from advanced_ba_project.data import ForestDataset, RoboflowTreeDataset, get_dataloaders
from advanced_ba_project.logger import log
from advanced_ba_project.model import UNet


def load_model(model_path, device):
    """Loads the trained U-Net model from models/ directory using the given model name."""
    model_path = Path(model_path)
    if not model_path.exists():
        log.error(f"Model file not found: {model_path}")
        exit(1)

    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    log.success(f"Loaded model from {model_path}")
    return model


def visualize_raw_and_predicted(
    model,
    dataset_name: str,
    data_path: Path,
    metadata_file: str = None,
    image_dir: Path = None,
    label_dir: Path = None,
    num_images: int = 5,
    img_dim: int = 256,
    seed: int = 42,
    indices: list = None,
    device="cpu"
):

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    target_transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor()
    ])

    # Load dataset
    if dataset_name == "forest":
        dataset = ForestDataset(
            data_path=data_path,
            metadata_file=metadata_file,
            transform=transform,
            target_transform=target_transform
        )
    elif dataset_name == "roboflow":
        dataset = RoboflowTreeDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            patch_size=img_dim,
            transform=transform,
            target_transform=target_transform
        )
    else:
        raise ValueError("dataset_name must be either 'forest' or 'roboflow'")

    # Pick indices
    if indices is None:
        random.seed(seed)
        indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
    else:
        num_images = len(indices)

    # Extract images/masks
    samples = [dataset[i] for i in indices]
    images = torch.stack([img for img, _ in samples]).to(device)
    masks = torch.stack([mask for _, mask in samples]).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        preds = model(images)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()

    # Plot
    fig, axs = plt.subplots(3, num_images, figsize=(4 * num_images, 12))

    for i in range(num_images):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * 0.5) + 0.5
        img = np.clip(img, 0, 1)

        gt_mask = masks[i].cpu().squeeze().numpy()
        pred_mask = preds[i].cpu().squeeze().numpy()

        forest_color = np.array([0, 1, 0])
        nonforest_color = np.array([1, 0.3, 0])
        alpha = 0.5

        # Ground Truth overlay
        overlay_gt = img.copy()
        mask_bool = gt_mask > 0.5
        overlay_gt[mask_bool] = (1 - alpha) * overlay_gt[mask_bool] + alpha * forest_color
        overlay_gt[~mask_bool] = (1 - alpha) * overlay_gt[~mask_bool] + alpha * nonforest_color

        # Prediction overlay
        overlay_pred = img.copy()
        mask_pred_bool = pred_mask > 0.5
        overlay_pred[mask_pred_bool] = (1 - alpha) * overlay_pred[mask_pred_bool] + alpha * forest_color
        overlay_pred[~mask_pred_bool] = (1 - alpha) * overlay_pred[~mask_pred_bool] + alpha * nonforest_color

        axs[0, i].imshow(img)
        axs[0, i].set_title(f"Original {indices[i]}")
        axs[1, i].imshow(overlay_gt)
        axs[1, i].set_title(f"Ground Truth {indices[i]}")
        axs[2, i].imshow(overlay_pred)
        axs[2, i].set_title(f"Prediction {indices[i]}")

        for ax in axs[:, i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # CLI Argument Parser
    model = load_model("models/unet_model_iconic-sweep-16.pth", device="cpu")

    visualize_raw_and_predicted(
        model=model,
        data_path=Path("data/raw/forest"),
        dataset_name="roboflow",
        image_dir=Path("data/raw/roboflow/train/images"),
        label_dir=Path("data/raw/roboflow/train/labelTxt"),
        num_images=5,
        img_dim=256,
        device="cpu"
    )
