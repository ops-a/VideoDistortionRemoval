import torch
import torch.nn as nn

class RainRemovalNet(nn.Module):
    def __init__(self):
        super(RainRemovalNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output normalized to [0, 1]
        )
    
    def forward(self, x):
        # Pass through encoder
        encoded = self.encoder(x)
        
        # Pass through bottleneck
        bottleneck = self.bottleneck(encoded)
        
        # Pass through decoder
        decoded = self.decoder(bottleneck)
        
        return decoded

# Example usage
if __name__ == "__main__":
    model = RainRemovalNet()
    print(model)
    # Example input: batch of 4 RGB images of size 128x128
    sample_input = torch.randn(4, 3, 128, 128)
    output = model(sample_input)
    print(f"Output shape: {output.shape}")