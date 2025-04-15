import random
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF


class Augmentation(Dataset):
    """Apply flips and rotations to images and masks."""
    
    def __init__(self, dataset, flip_prob=0.5, rotate_prob=0.3):
        self.dataset = dataset
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        image, mask = self.dataset[idx]
        

        if random.random() < self.flip_prob:
            image = torch.flip(image, [2]) 
            mask = torch.flip(mask, [2])
        
        # Random vertical flip
        if random.random() < self.flip_prob:
            image = torch.flip(image, [1])  
            mask = torch.flip(mask, [1])
        
        # Random rotation (90, 180, or 270 degrees)
        if random.random() < self.rotate_prob:
            k = random.choice([1, 2, 3]) 
            image = torch.rot90(image, k, [1, 2]) 
            mask = torch.rot90(mask, k, [1, 2])
        
        return image, mask

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


class RoboflowTreeDataset(Dataset):
    """Dataset for Roboflow-style images + polygon txt labels, extracting only 'Tree' annotations."""

    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        patch_size: int = 256,
        transform=None,
        target_transform=None,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.transform = transform
        self.target_transform = target_transform

        self.samples = self._build_samples()

    def _build_samples(self) -> List[Tuple[Path, Path, Tuple[int, int]]]:
        samples = []
        for image_path in sorted(self.image_dir.glob("*.jpg")):
            label_path = self.label_dir / (image_path.stem + ".txt")
            if not label_path.exists():
                continue

            for y in range(0, 512, self.patch_size):
                for x in range(0, 512, self.patch_size):
                    samples.append((image_path, label_path, (x, y)))
        return samples

    def _parse_tree_polygons(self, label_path: Path) -> List[List[Tuple[float, float]]]:
        polygons = []
        with open(label_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue  # Not a valid polygon line
                label = parts[-2]
                if label.lower() != "tree":
                    continue
                coords = list(map(float, parts[:-2]))
                polygon = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                polygons.append(polygon)
        return polygons

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_path, (x, y) = self.samples[idx]

        image = Image.open(image_path).convert("RGB").crop((x, y, x + self.patch_size, y + self.patch_size))

        # Build mask
        mask = Image.new("L", (512, 512), 0)
        polygons = self._parse_tree_polygons(label_path)
        for poly in polygons:
            ImageDraw.Draw(mask).polygon(poly, outline=255, fill=255)
        mask = mask.crop((x, y, x + self.patch_size, y + self.patch_size))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


def get_dataloaders(
    data_path: Path,
    metadata_file: str,
    roboflow_train_path: Path,
    roboflow_val_path: Path,
    roboflow_test_path: Path,
    batch_size: int = 32,
    img_dim: int = 256,
    train_ratio: float = 0.80,
    seed: int = 42,
    subset: bool = False,
    apply_augmentation: bool = False,
):
    
    """Creates combined train and validation dataloaders from Forest and Roboflow datasets."""

    transform = transforms.Compose(
        [
            transforms.Resize((img_dim, img_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    target_transform = transforms.Compose(
        [
            transforms.Resize((img_dim, img_dim)),
            transforms.ToTensor(),
        ]
    )

    # Load Forest Dataset
    forest_dataset = ForestDataset(data_path, metadata_file, transform=transform, target_transform=target_transform)

    # Reduce dataset size for quick testing
    if subset:
        torch.manual_seed(seed)
        indices = random.sample(range(len(forest_dataset)), min(len(forest_dataset), 120))
        forest_dataset = Subset(forest_dataset, indices)

    # Load Roboflow Datasets (already pre-split into train/val)
    roboflow_train = RoboflowTreeDataset(
        image_dir=roboflow_train_path / "images",
        label_dir=roboflow_train_path / "labelTxt",
        patch_size=img_dim,
        transform=transform,
        target_transform=target_transform,
    )

    roboflow_val = RoboflowTreeDataset(
        image_dir=roboflow_val_path / "images",
        label_dir=roboflow_val_path / "labelTxt",
        patch_size=img_dim,
        transform=transform,
        target_transform=target_transform,
    )

    roboflow_test = RoboflowTreeDataset(
        image_dir=roboflow_test_path / "images",
        label_dir=roboflow_test_path / "labelTxt",
        patch_size=img_dim,
        transform=transform,
        target_transform=target_transform,
    )

    # Split Forest into train/val/test
    train_size = int(train_ratio * len(forest_dataset))
    val_size = int(0.1 * len(forest_dataset))
    test_size = len(forest_dataset) - train_size - val_size
    forest_train, forest_val, forest_test = random_split(forest_dataset, [train_size, val_size, test_size])

    # Combine datasets
    combined_train = ConcatDataset([forest_train, roboflow_train])
    combined_val = ConcatDataset([forest_val, roboflow_val])
    combined_test = ConcatDataset([forest_test, roboflow_test])

    # Shuffle training set once
    torch.manual_seed(seed)
    train_indices = torch.randperm(len(combined_train)).tolist()
    combined_train = Subset(combined_train, train_indices)
    
    # Apply augmentations only to training data
    if apply_augmentation:
        combined_train = Augmentation(combined_train)

    # Shuffle validation set once (different seed for variation)
    torch.manual_seed(seed + 1)
    val_indices = torch.randperm(len(combined_val)).tolist()
    combined_val = Subset(combined_val, val_indices)

    # Shuffle test set once (different seed for variation)
    torch.manual_seed(seed + 2)
    test_indices = torch.randperm(len(combined_test)).tolist()
    combined_test = Subset(combined_test, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(combined_val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(combined_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

def visualize_raw_data(
    dataset_name: str,
    data_path: Path,
    metadata_file: str = None,
    image_dir: Path = None,
    label_dir: Path = None,
    num_images: int = 5,
    img_dim: int = 256,
    seed: int = 42,
    indices: list = None
):
    """
    Visualize satellite images and forest segmentation masks from a specified dataset.

    Args:
        dataset_name (str): Either 'forest' or 'roboflow'.
        data_path (Path): Base path to dataset root.
        metadata_file (str): Required if dataset is 'forest' (name of metadata CSV).
        image_dir (Path): Required if dataset is 'roboflow'.
        label_dir (Path): Required if dataset is 'roboflow'.
        num_images (int): Number of images to show (ignored if indices is provided).
        img_dim (int): Target resolution (square).
        seed (int): Seed for reproducible sampling.
        indices (list[int], optional): Specific indices to visualize from dataset.
    """
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

    # Select dataset
    if dataset_name == "forest":
        if metadata_file is None:
            raise ValueError("metadata_file must be provided for forest dataset.")
        dataset = ForestDataset(
            data_path=data_path,
            metadata_file=metadata_file,
            transform=transform,
            target_transform=target_transform
        )
    elif dataset_name == "roboflow":
        if image_dir is None or label_dir is None:
            raise ValueError("image_dir and label_dir must be provided for roboflow dataset.")
        dataset = RoboflowTreeDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            patch_size=img_dim,
            transform=transform,
            target_transform=target_transform
        )
    else:
        raise ValueError("dataset_name must be either 'forest' or 'roboflow'.")

    # Get sample indices
    if indices is None:
        torch.manual_seed(seed)
        random.seed(seed)
        indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
    else:
        num_images = len(indices)

    samples = [dataset[i] for i in indices]

    # Plot
    fig, axs = plt.subplots(2, num_images, figsize=(4 * num_images, 8))

    for i, (image, mask) in enumerate(samples):
        image = image.permute(1, 2, 0).numpy()
        image = (image * 0.5) + 0.5
        image = np.clip(image, 0, 1)

        mask = mask.squeeze().numpy()
        if mask.ndim == 3:
            mask = mask[0]

        axs[0, i].imshow(image)
        axs[0, i].axis("off")
        axs[0, i].set_title(f"{dataset_name.capitalize()} Original {indices[i]}")

        overlay = image.copy()
        forest_color = np.array([0, 1, 0])
        nonforest_color = np.array([1, 0.3, 0])
        alpha = 0.5

        mask_bool = mask > 0.5
        overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * forest_color
        overlay[~mask_bool] = (1 - alpha) * overlay[~mask_bool] + alpha * nonforest_color

        axs[1, i].imshow(overlay)
        axs[1, i].axis("off")
        axs[1, i].set_title(f"{dataset_name.capitalize()} With Mask {indices[i]}")

    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Forest', alpha=0.5),
        Patch(facecolor='orangered', edgecolor='black', label='Non-Forest', alpha=0.5),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize='large')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def summarize_raw_data(dataset_name: str, data_path: Path = None, metadata_file: str = None,
                          image_dir: Path = None, label_dir: Path = None, img_dim: int = 256):
    """
    Summarize a raw dataset: number of images and percentage with forest labels.

    Args:
        dataset_name (str): 'forest' or 'roboflow'.
        data_path (Path): Required for forest dataset.
        metadata_file (str): Required for forest dataset.
        image_dir (Path): Required for roboflow dataset.
        label_dir (Path): Required for roboflow dataset.
        img_dim (int): Image patch size.
    """
    transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
    ])

    if dataset_name == "forest":
        if not data_path or not metadata_file:
            raise ValueError("For 'forest' dataset, provide data_path and metadata_file.")
        dataset = ForestDataset(data_path, metadata_file, transform=transform, target_transform=transform)

    elif dataset_name == "roboflow":
        if not image_dir or not label_dir:
            raise ValueError("For 'roboflow' dataset, provide image_dir and label_dir.")
        dataset = RoboflowTreeDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            patch_size=img_dim,
            transform=transform,
            target_transform=transform
        )
    else:
        raise ValueError("dataset_name must be either 'forest' or 'roboflow'.")

    total = len(dataset)
    forest = 0

    for i in tqdm(range(total)):
        _, mask = dataset[i]
        if mask.sum() > 0:
            forest += 1

    percent = (forest / total) * 100
    print(f"Dataset: {dataset_name}")
    print(f"Total samples: {total}")
    print(f"Amount of samples with forest: {forest} ({percent:.2f}%)")




if __name__ == "__main__":
    data_path = Path("data/raw/Forest Segmented")
    metadata_file = "meta_data.csv"
    roboflow_train_path = Path("data/raw/roboflow/train")
    roboflow_val_path = Path("data/raw/roboflow/valid")
    roboflow_test_path = Path("data/raw/roboflow/test")

    # visualize_raw_data(
    #     dataset_name="forest",
    #     data_path=Path("data/raw/Forest Segmented"),
    #     metadata_file="meta_data.csv",
    #     indices=[0, 3, 7, 12, 25]
    # )

    # visualize_raw_data(
    #     dataset_name="roboflow",
    #     data_path=None,
    #     image_dir=Path("data/raw/roboflow/train/images"),
    #     label_dir=Path("data/raw/roboflow/train/labelTxt"),
    #     indices=[0,2,3,4,5]
    # )

    summarize_raw_data(
    dataset_name="forest",
    data_path=data_path,
    metadata_file=metadata_file
    )

    summarize_raw_data(
        dataset_name="roboflow",
        image_dir=roboflow_train_path / "images",
        label_dir=roboflow_train_path / "labelTxt"
    )

    # def count_empty_masks(dataloader, name=""):
    #     empty = 0
    #     total = 0
    #     for images, masks in dataloader:
    #         batch_size = masks.size(0)
    #         total += batch_size
    #         # Flatten and sum each mask: if sum == 0, it's an empty mask
    #         empty += (masks.view(batch_size, -1).sum(dim=1) == 0).sum().item()

    #     print(f"[{name}] Empty masks: {empty} / {total} ({(empty / total) * 100:.2f}%)")

    # # Count empty masks
    # count_empty_masks(train_loader, name="Train")
    # count_empty_masks(val_loader, name="Val")
    # count_empty_masks(test_loader, name="Test")
