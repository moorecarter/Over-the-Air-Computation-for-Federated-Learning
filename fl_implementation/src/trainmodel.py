import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image


# ── Data ──────────────────────────────────────────────────────────────

class GTSRBDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_train_data(data_dir):
    paths = []
    labels = []
    train_dir = Path(data_dir) / "Train"
    for folder in train_dir.iterdir():
        if folder.is_dir():
            label = int(folder.name)
            for image in folder.glob("*.png"):
                paths.append(image)
                labels.append(label)
    return paths, labels


def load_test_data(data_dir):
    paths = []
    labels = []
    data_dir = Path(data_dir)
    with open(data_dir / "Test.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(data_dir / row["Path"])
            labels.append(int(row["ClassId"]))
    return paths, labels


def get_transforms(split, img_size=32):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# ── Model ─────────────────────────────────────────────────────────────

class SmallCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Training ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────

def main():
    data_dir = "../../dataset"
    img_size = 32
    batch_size = 64
    num_epochs = 5
    lr = 1e-3
    num_classes = 43

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data
    train_paths, train_labels = load_train_data(data_dir)
    test_paths, test_labels = load_test_data(data_dir)
    print(f"Train: {len(train_paths)} images, Test: {len(test_paths)} images, Classes: {num_classes}")

    # Create datasets
    full_train = GTSRBDataset(train_paths, train_labels, get_transforms("train", img_size))
    test_dataset = GTSRBDataset(test_paths, test_labels, get_transforms("test", img_size))

    # Split train into train/val (90/10)
    val_size = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    model = SmallCNN(num_classes=num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Baseline (no training)
    base_loss, base_acc = evaluate(model, test_loader, criterion, device)
    print(f"Before training | Test: loss={base_loss:.4f} acc={base_acc*100:.1f}%\n")

    # Train
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc*100:.1f}% | "
              f"Val: loss={val_loss:.4f} acc={val_acc*100:.1f}%")

    # Test
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc*100:.1f}%")

    # Save
    torch.save(model.state_dict(), "gtsrb_cnn.pth")
    print("Model saved to gtsrb_cnn.pth")


if __name__ == "__main__":
    main()
