
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from src.model import CIFAR10_CNN



#  LOAD DATASET


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


# Image transformations

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# Load CIFAR-10 test dataset

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

classes = test_dataset.classes

print("Classes:", classes)
print("Test images:", len(test_dataset))



#  LOAD TRAINED MODEL


model = CIFAR10_CNN().to(device)

model.load_state_dict(
    torch.load(
        "models/best_cifar10_cnn.pth",
        map_location=device
    )
)

model.eval()

print("Best model loaded successfully!")



# MODEL EVALUATION

test_correct = 0
test_total = 0

all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        test_total += labels.size(0)

        test_correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


test_accuracy = 100 * test_correct / test_total

print("Test Accuracy:", round(test_accuracy, 2), "%")


# PHASE 4: CONFUSION MATRIX


cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("CIFAR-10 Confusion Matrix")

plt.tight_layout()
plt.show()



# CLASSIFICATION REPORT


print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes
    )
)


print("========== FINAL MODEL PERFORMANCE ==========")

print("Model: CNN from Scratch")
print("Dataset: CIFAR-10")

print("Number of Classes:", len(classes))
print("Testing Images:", len(test_dataset))

print("Final Test Accuracy:", round(test_accuracy, 2), "%")
