from torch.utils.data import Dataset
import torch
from advanced_ba_project.data import get_dataloaders


def test_get_dataloaders():
    # Test the get_dataloaders function
    train_loader, val_loader = get_dataloaders(batch_size=8, num_workers=0)
    assert len(train_loader) > 0
    assert len(val_loader) > 0

    # Check if the dataloaders return instances of DataLoader
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)

    # Check if the dataloaders return instances of Dataset
    assert isinstance(train_loader.dataset, Dataset)
    assert isinstance(val_loader.dataset, Dataset)

    # Check if the dataloaders return the correct batch size
    assert train_loader.batch_size == 8
    assert val_loader.batch_size == 8



