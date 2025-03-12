import torch

from advanced_ba_project.model import UNet


def test_unet():
    # Test the UNet model
    model = UNet(in_channels=3, out_channels=1)
    assert model is not None
    assert model.__class__.__name__ == "UNet"
    assert model.in_channels == 3
    assert model.out_channels == 1
    assert model.encoder is not None
    assert model.decoder is not None

    # Test the forward pass
    x = torch.randn((1, 3, 128, 128))
    out = model(x)
    assert out.shape == (1, 1, 128, 128)
    print("UNet model test passed.")
