import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from advanced_ba_project.data import get_dataloaders


class MockDataset(Dataset):
    """A fake dataset for testing the dataloader without requiring actual data."""

    def __init__(self, num_samples=100, img_size=(3, 256, 256)):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(*self.img_size)  # Fake image tensor
        mask = torch.randint(0, 2, (1, 256, 256))  # Fake binary mask
        return image, mask


@pytest.fixture
def mock_dataloader():
    """Fixture to create a dataloader with a mocked dataset."""
    dataset = MockDataset(num_samples=50)  # Fake dataset with 50 samples
    return DataLoader(dataset, batch_size=8, shuffle=True)


def test_mock_dataloader(mock_dataloader):
    """Test that the mock dataloader correctly loads fake data."""
    batch = next(iter(mock_dataloader))  # Get first batch
    images, masks = batch

    # Check shapes
    assert images.shape == (8, 3, 256, 256)
    assert masks.shape == (8, 1, 256, 256)

    # Check that it's a PyTorch tensor
    assert isinstance(images, torch.Tensor)
    assert isinstance(masks, torch.Tensor)
