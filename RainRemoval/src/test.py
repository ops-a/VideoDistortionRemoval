import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from models.model import RainRemovalNet

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

# Main Testing Script
if __name__ == "__main__":
    # Paths
    test_dir = "./datasets/test_a"  # Change to test_b for the other test set
    model_path = "./models/rain_removal_net.pth"
    output_dir = "./outputs/test_a"  # Change to test_b for the other test set

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    test_dataset = TestRainDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load the trained model
    model = RainRemovalNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    # Test the model
    test_model(model, test_loader, device, output_dir)
    print(f"Cleaned images saved to {output_dir}")