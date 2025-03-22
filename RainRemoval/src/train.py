import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from models.model import RainRemovalNet

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

# Training Function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

# Main Training Script
if __name__ == "__main__":
    # Paths
    train_dir = "/home/redstar/Workspace/VideoDistortionRemoval/RainRemoval/datasets/train"

    # Hyperparameters
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 20

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    train_dataset = RainDataset(train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, and Optimizer
    model = RainRemovalNet().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, num_epochs)

    # Save the trained model
    model_save_path = "/home/redstar/Workspace/VideoDistortionRemoval/RainRemoval/models/rain_removal_net.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")