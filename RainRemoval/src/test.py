import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from models.model import AdvancedRainRemovalNet

# Custom Dataset for Testing
class TestRainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_dir = os.path.join(root_dir, "data")
        self.data_files = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        data_image = Image.open(data_path).convert("RGB")

        if self.transform:
            data_image = self.transform(data_image)

        return data_image, self.data_files[idx]  # Return image and filename

# Function to Test the Model
def test_model(model, test_loader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for inputs, filenames in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Save the output images
            outputs = outputs.cpu().permute(0, 2, 3, 1).numpy()  # Convert to (batch, height, width, channels)
            for i, filename in enumerate(filenames):
                output_image = (outputs[i] * 255).astype("uint8")  # Convert to uint8
                output_image = Image.fromarray(output_image)
                output_image.save(os.path.join(output_dir, filename))
                print(f"Saved cleaned image: {filename}")

# Main Testing Script
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the Advanced Rain Removal model.")
    parser.add_argument("--test_dir", type=str, default="./datasets/test_a", help="Path to the test dataset.")
    parser.add_argument("--model_path", type=str, default="./models/advanced_rain_removal_net.pth", help="Path to the trained model.")
    parser.add_argument("--output_dir", type=str, default="./outputs/test_a", help="Path to save the cleaned images.")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the input images (default: 256).")
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    test_dataset = TestRainDataset(args.test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load the trained model
    model = AdvancedRainRemovalNet(image_size=args.image_size).to(device)  # Pass image_size to the model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model from {args.model_path}")

    # Test the model
    test_model(model, test_loader, device, args.output_dir)
    print(f"Cleaned images saved to {args.output_dir}")