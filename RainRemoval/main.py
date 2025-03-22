import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.train import RainDataset, train_model  # Import dataset and training logic
from models.model import RainRemovalNet  # Import the model

if __name__ == "__main__":
    # Paths
    train_dir = "./datasets/train"

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
    model_save_path = "./models/rain_removal_net.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")