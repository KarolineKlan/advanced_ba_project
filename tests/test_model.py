import torch
from advanced_ba_project.model import UNet

def test_unet_forward():
    # Initialize the model
    model = UNet(in_channels=3, out_channels=1)

    # Create dummy input
    x = torch.randn((1, 3, 256, 256))  # One batch, 3 channels, 256x256 image

    # Forward pass
    output = model(x)

    # Check that the output shape matches expectation
    assert output.shape == (1, 1, 256, 256), "Output shape mismatch"

def test_unet_output_no_nans():
    # Initialize the model
    model = UNet(in_channels=3, out_channels=1)

    # Dummy input
    x = torch.randn((1, 3, 256, 256))

    # Forward pass
    output = model(x)

    # Check that there are no NaNs in the output
    assert not torch.isnan(output).any(), "Output contains NaNs"
