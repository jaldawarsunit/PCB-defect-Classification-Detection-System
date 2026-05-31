import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import os

DATA_DIR = "dataset_split"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=32)

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 6)
)

model.load_state_dict(torch.load("pcb_defect_model.pth", map_location=DEVICE))

model = model.to(DEVICE)

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")