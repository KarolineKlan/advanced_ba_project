import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path (Encoder)
        for feature in features:
            self.encoder.append(self.contracting_block(in_channels, feature))
            in_channels = feature

        # Bottom layer (Bottleneck)
        self.bottleneck = self.contracting_block(features[-1], features[-1] * 2)

        # Expanding path (Decoder)
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))  # Upsample
            self.decoder.append(self.contracting_block(feature * 2, feature))  # Double channels due to skip connection

        # Final output layer
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for enc_layer in self.encoder:
            x = enc_layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]  # Reverse the skip connections
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # Up-convolution
            skip_connection = skip_connections[i // 2]
            x = torch.cat((skip_connection, x), dim=1)  # Concatenate along the channel dimension
            x = self.decoder[i + 1](x)  # Pass through conv layers

        # Final layer
        x = self.final_layer(x)

        return x  # Raw logits; apply Sigmoid if using BCEWithLogitsLoss

    def contracting_block(self, in_channels, out_channels):
        """Creates a block with two convolutional layers followed by BatchNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


# Test the model
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)  # 3-channel input, 1-channel output
    x = torch.randn(1, 3, 256, 256)  # Example input (batch_size=1, RGB image 256x256)
    preds = model(x)
    print(f"Output shape: {preds.shape}")  # Should be (1, 1, 256, 256)
