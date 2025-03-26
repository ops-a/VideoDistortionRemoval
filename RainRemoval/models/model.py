import torch
import torch.nn as nn

# Squeeze-and-Excitation Block for Attention
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = x.view(batch, channels, -1).mean(dim=2)  # Global Average Pooling
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        return x * y

# Advanced Rain Removal Network
class AdvancedRainRemovalNet(nn.Module):
    def __init__(self):
        super(AdvancedRainRemovalNet, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck with SEBlock
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(512),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output normalized to [0, 1]
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc3)
        
        # Decoder with skip connections
        dec3 = self.decoder3(bottleneck + enc3)  # Skip connection
        dec2 = self.decoder2(dec3 + enc2)        # Skip connection
        dec1 = self.decoder1(dec2 + enc1)        # Skip connection
        
        return dec1

# Example usage
if __name__ == "__main__":
    model = AdvancedRainRemovalNet()
    print(model)
    # Example input: batch of 4 RGB images of size 128x128
    sample_input = torch.randn(4, 3, 256, 256)
    output = model(sample_input)
    print(f"Output shape: {output.shape}")