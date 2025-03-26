import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.train import RainDataset, train_model, PerceptualLoss  # Import dataset, training logic, and perceptual loss
from models.model import AdvancedRainRemovalNet  # Import the advanced model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the Advanced Rain Removal model.")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the input images (default: 256).")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training (default: 8).")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training (default: 0.0001).")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Paths
    train_dir = "./datasets/train"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    train_dataset = RainDataset(train_dir, transform=transform)
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
    model_save_path = "./models/advanced_rain_removal_net.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")