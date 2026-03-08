import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from medmnist import BloodMNIST


# ── Data ──────────────────────────────────────────────────────────────

class BloodMNISTWrapper(Dataset):
    def __init__(self, split="train", img_size=28, transform=None):
        self.dataset = BloodMNIST(split=split, download=True, size=img_size, as_rgb=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(label.squeeze(), dtype=torch.long)
        return img, label


def get_transforms(split, img_size=28):
    if split == "train":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# ── Model ─────────────────────────────────────────────────────────────

class SmallCNN(nn.Module):
    def __init__(self, num_classes=8, img_size=28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # After 3 MaxPool2d(2): img_size // 8
        flat_size = 128 * (img_size // 8) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
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
    img_size = 28
    batch_size = 64
    num_epochs = 15
    lr = 1e-4
    num_classes = 8

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data
    train_dataset = BloodMNISTWrapper("train", img_size, get_transforms("train", img_size))
    val_dataset = BloodMNISTWrapper("val", img_size, get_transforms("test", img_size))
    test_dataset = BloodMNISTWrapper("test", img_size, get_transforms("test", img_size))
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}, Classes: {num_classes}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    model = SmallCNN(num_classes=num_classes, img_size=img_size).to(device)
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
    # torch.save(model.state_dict(), "bloodmnist_cnn.pth")
    # print("Model saved to bloodmnist_cnn.pth")


if __name__ == "__main__":
    main()
