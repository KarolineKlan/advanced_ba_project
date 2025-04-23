import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=64, dropout_rate=0.2):
        super().__init__()
        features = init_features
        self.dropout_rate = dropout_rate

        # Contracting path
        self.enc1 = self.contracting_block(in_channels, features)
        self.enc2 = self.contracting_block(features, features * 2)
        self.enc3 = self.contracting_block(features * 2, features * 4)
        self.enc4 = self.contracting_block(features * 4, features * 8)

        # Bottom
        self.bottleneck = self.bottom_layer(features * 8, features * 16)

        # Expanding path
        self.dec1 = self.expanding_block(features * 16 + features * 8, features * 8)
        self.dec2 = self.expanding_block(features * 8 + features * 4, features * 4)
        self.dec3 = self.expanding_block(features * 4 + features * 2, features * 2)
        self.dec4 = self.expanding_block(features * 2 + features, features)

        # Final layer
        self.final = nn.Conv2d(features, out_channels, kernel_size=1)

        # Pool and upsample ops
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        # Bottleneck
        x_bot = self.bottleneck(self.pool(x4))

        # Decoder with skip connections
        x = self.upsample_concat(x_bot, x4)
        x = self.dec1(x)

        x = self.upsample_concat(x, x3)
        x = self.dec2(x)

        x = self.upsample_concat(x, x2)
        x = self.dec3(x)

        x = self.upsample_concat(x, x1)
        x = self.dec4(x)

        return self.final(x)

    def contracting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def bottom_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def expanding_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def upsample_concat(self, x1, x2):
        upsampled = nn.functional.interpolate(x1, scale_factor=2, mode="bilinear", align_corners=True)
        return torch.cat([upsampled, x2], dim=1)


# Optional quick test
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1, dropout_rate=0.2)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Output shape: {y.shape}")  # Expect (1, 1, 256, 256)
