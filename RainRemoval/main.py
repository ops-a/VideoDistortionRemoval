import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.train import RainDataset, train_model, PerceptualLoss  # Import dataset, training logic, and perceptual loss
from models.model import AdvancedRainRemovalNet  # Import the advanced model

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
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    train_dataset = RainDataset(train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, and Optimizer
    model = AdvancedRainRemovalNet().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    perceptual_loss = PerceptualLoss().to(device)  # Perceptual Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs

    # Train the model
    train_model(model, train_loader, criterion, perceptual_loss, optimizer, scheduler, device, num_epochs)

    # Save the trained model
    model_save_path = "./models/advanced_rain_removal_net.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")