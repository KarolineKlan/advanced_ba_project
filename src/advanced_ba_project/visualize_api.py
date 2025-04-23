import argparse
import datetime
import glob
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
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

def load_tif_image(path, img_dim=256):
    # Open and handle .tif with potential multi-band issues
    img = Image.open(path)
    img_array = np.array(img)

    # If more than 3 channels, keep the first 3:
    if img_array.ndim == 3 and img_array.shape[2] > 3:
        img_array = img_array[:, :, :3]

    # If single-channel grayscale, stack it to fake RGB:
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    # Scale pixel values to 0-1 (float32)
    img_array = img_array.astype(np.float32)
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)

    # Convert back to PIL Image for transforms
    img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
    mean = np.array([0.3448, 0.3311, 0.2385])
    std = np.array([0.1063, 0.0827, 0.0723])
    # Apply same transforms as in training
    transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform(img_pil)

def predict_on_api_images(
    model,
    image_folder: Path,
    indices: list = None,
    num_images: int = 5,
    img_dim: int = 256,
    device="cpu"
):
    # Find all .tif images in the folder
    image_paths = list(image_folder.glob("*.tif"))
    if len(image_paths) == 0:
        raise ValueError(f"No .tif images found in {image_folder}")

    # Select paths based on given indices or sample randomly
    if indices is not None:
        selected_paths = [image_paths[i] for i in indices]
    else:
        selected_paths = random.sample(image_paths, min(num_images, len(image_paths)))

    # Load images and apply transform
    images = []
    for path in selected_paths:
        img_tensor = load_tif_image(path, img_dim)
        images.append(img_tensor)

    images = torch.stack(images).to(device)

    # Predict with model
    model.eval()
    with torch.no_grad():
        preds = model(images)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()

    # Plot predictions
    fig, axs = plt.subplots(2, len(selected_paths), figsize=(4 * len(selected_paths), 8))
    for i in range(len(selected_paths)):
        # Undo normalization using true mean and std:
        mean = np.array([0.3448, 0.3311, 0.2385])
        std = np.array([0.1063, 0.0827, 0.0723])

        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * std) + mean
        img = np.clip(img, 0, 1)

        pred_mask = preds[i].cpu().squeeze().numpy()

        forest_color = np.array([0, 1, 0])
        nonforest_color = np.array([1, 0.3, 0])
        alpha = 0.5

        # Prediction overlay
        overlay_pred = img.copy()
        mask_pred_bool = pred_mask > 0.5
        overlay_pred[mask_pred_bool] = (1 - alpha) * overlay_pred[mask_pred_bool] + alpha * forest_color
        overlay_pred[~mask_pred_bool] = (1 - alpha) * overlay_pred[~mask_pred_bool] + alpha * nonforest_color

        axs[0, i].imshow(img)
        axs[0, i].set_title(f"Original {selected_paths[i].name}")
        axs[0, i].axis("off")

        axs[1, i].imshow(overlay_pred)
        axs[1, i].set_title("Prediction")
        axs[1, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # CLI Argument Parser
    model = load_model("models/unet_model_2025-04-16_11-07-12.pth", device="cpu")

    # visualize_raw_and_predicted(
    #     model=model,
    #     data_path=Path("data/raw/forest"),
    #     dataset_name="roboflow",
    #     image_dir=Path("data/raw/roboflow/test/images"),
    #     label_dir=Path("data/raw/roboflow/test/labelTxt"),
    #     num_images=5,
    #     img_dim=256,
    #     device="cpu",
    #     indices=[10, 20, 30, 40, 50]  # Example indices to visualize
    # )

    predict_on_api_images(
        model=model,
        image_folder=Path("data/Deforestation_pics"),
        num_images=5,
        img_dim=256,
        device="cpu",
        indices=[6,7,8,9]  # Example indices to visualize
    )
