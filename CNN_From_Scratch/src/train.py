
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from pathlib import Path
from torch.utils.data import DataLoader, random_split
from src.model import CIFAR10_CNN


# ==========================================
# PHASE 1: DATASET PREPARATION
# ==========================================

BATCH_SIZE = 128
RANDOM_SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


# Image transformations

train_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# Load CIFAR-10 training dataset

train_raw = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)


# Split into training and validation

train_indices, val_indices = random_split(
    range(len(train_raw)),
    [45000, 5000],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)


# Create datasets with different transformations

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=train_transform
)

val_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=test_transform
)


# Apply split indices

train_dataset = torch.utils.data.Subset(
    train_dataset,
    train_indices.indices
)

val_dataset = torch.utils.data.Subset(
    val_dataset,
    val_indices.indices
)


# Create DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ==========================================
# PHASE 2: MODEL SETUP
# ==========================================

model = CIFAR10_CNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0005
)


# Learning rate scheduler

num_epochs = 30

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
)


# ==========================================
# PHASE 3: TRAINING + EARLY STOPPING
# ==========================================

patience = 5
counter = 0

best_val_accuracy = 0.0

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


for epoch in range(num_epochs):

    # Training

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total


    # Validation

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / val_total
    val_accuracy = 100 * val_correct / val_total


    # Store metrics

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)


    # Save best model

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        counter = 0

        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        torch.save(
            model.state_dict(),
            model_dir / "best_cifar10_cnn.pth"
        )

        print("Best model saved!")

    else:

        counter += 1


    scheduler.step()


    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}% "
        f"Counter: {counter}/{patience}"
    )


    # Early stopping

    if counter >= patience:

        print("Early stopping triggered!")
        break


print("\nBest Validation Accuracy:", round(best_val_accuracy, 2), "%")


# ==========================================
# PHASE 4: SAVE TRAINED MODEL
# ==========================================

model.load_state_dict(
    torch.load(
        "models/best_cifar10_cnn.pth",
        map_location=device
    )
)

model.eval()

print("Best model restored!")

print("Model saved successfully!")
