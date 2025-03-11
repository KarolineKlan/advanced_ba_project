from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class ForestDataset(Dataset):
    """Custom dataset for loading satellite images of forests and their segmentation masks."""

    def __init__(self, data_path: Path, metadata_file: str, transform=None, target_transform=None) -> None:
        """
        Args:
            data_path (Path): Path to the dataset containing 'images/' and 'masks/' directories.
            metadata_file (str): Filename of the metadata CSV file.
            transform: Transformations to apply to the images.
            target_transform: Transformations to apply to the masks.
        """
        self.data_path = data_path
        self.image_dir = data_path / "images"
        self.mask_dir = data_path / "masks"
        self.transform = transform
        self.target_transform = target_transform

        # Load metadata CSV
        metadata_path = data_path / metadata_file
        self.metadata = pd.read_csv(metadata_path)

        # Ensure necessary columns exist
        if not {"image", "mask"}.issubset(self.metadata.columns):
            raise ValueError("Metadata CSV must contain 'image' and 'mask' columns.")

        # Get image and mask paths
        self.image_paths = [self.image_dir / fname for fname in self.metadata["image"]]
        self.mask_paths = [self.mask_dir / fname for fname in self.metadata["mask"]]

        assert len(self.image_paths) == len(self.mask_paths), "Mismatch in images and masks count!"

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single sample from the dataset."""
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # Open images
        image = Image.open(img_path).convert("RGB")  # Convert image to RGB
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


def get_dataloaders(
    data_path: Path, 
    metadata_file: str, 
    batch_size: int = 32, 
    img_dim: int = 256, 
    train_ratio: float = 0.85, 
    seed: int = 42
):
    """Creates train and validation dataloaders by splitting the dataset."""
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize between -1 and 1
    ])

    target_transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
    ])

    # Load full dataset
    full_dataset = ForestDataset(data_path, metadata_file, transform=transform, target_transform=target_transform)

    # Split dataset into train and validation
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    # Ensure reproducibility
    torch.manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Total samples: {total_size} | Train: {train_size} | Val: {val_size}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    data_path = Path("data/raw/Forest Segmented")
    metadata_file = "meta_data.csv"

    train_loader, val_loader = get_dataloaders(data_path, metadata_file)

    # Example: Fetch a batch
    for images, masks in train_loader:
        print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")  # Should be (batch_size, 3, 256, 256) and (batch_size, 1, 256, 256)
        break
    
    
    # Get a batch
    images, masks = next(iter(train_loader))

    # Convert tensors to numpy format for plotting
    image_np = images[0].permute(1, 2, 0).numpy()  # Change shape to (H, W, C)
    mask_np = masks[0].squeeze(0).numpy()  # Remove the single color channel

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow((image_np * 0.5) + 0.5)  # Undo normalization for visualization
    ax[0].set_title("Satellite Image")
    ax[0].axis("off")

    ax[1].imshow(mask_np, cmap="gray")
    ax[1].set_title("Segmentation Mask")
    ax[1].axis("off")

    # Show the plot
    plt.show()