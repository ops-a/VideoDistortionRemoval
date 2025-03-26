import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from models.model import AdvancedRainRemovalNet

# Custom Dataset
class RainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_dir = os.path.join(root_dir, "data")
        self.gt_dir = os.path.join(root_dir, "gt")
        self.data_files = sorted(os.listdir(self.data_dir))
        self.gt_files = sorted(os.listdir(self.gt_dir))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])

        data_image = Image.open(data_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")

        if self.transform:
            data_image = self.transform(data_image)
            gt_image = self.transform(gt_image)

        return data_image, gt_image

# Perceptual Loss using VGG
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights="VGG16_Weights.DEFAULT").features[:16]  # Use first few layers of VGG16
        self.vgg = vgg.eval()  # Set to evaluation mode
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG parameters
        self.criterion = nn.MSELoss()

    def forward(self, output, target):
        output_features = self.vgg(output)
        target_features = self.vgg(target)
        return self.criterion(output_features, target_features)

# Training Function
def train_model(model, train_loader, criterion, perceptual_loss, optimizer, scheduler, device, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets) + 0.1 * perceptual_loss(outputs, targets)  # Combine losses

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the Advanced Rain Removal model.")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the input images (default: 256).")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training (default: 8).")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training (default: 0.0001).")
    parser.add_argument("--train_dir", type=str, default="./datasets/train", help="Path to the training dataset.")
    parser.add_argument("--model_save_path", type=str, default="./models/advanced_rain_removal_net.pth", help="Path to save the trained model.")
    return parser.parse_args()

# Main Training Script
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    train_dataset = RainDataset(args.train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Model, Loss, and Optimizer
    model = AdvancedRainRemovalNet(image_size=args.image_size).to(device)  # Pass image_size to the model
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    perceptual_loss = PerceptualLoss().to(device)  # Perceptual Loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs

    # Train the model
    train_model(model, train_loader, criterion, perceptual_loss, optimizer, scheduler, device, args.epochs)

    # Save the trained model
    torch.save(model.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")