
import torch
import torchvision
import torchvision.transforms as transforms
import gradio as gr

from src.model import CIFAR10_CNN


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# Load CIFAR-10 test dataset to get class names
test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True
)

classes = test_dataset.classes


# Load trained model
model = CIFAR10_CNN().to(device)

model.load_state_dict(
    torch.load(
        "models/best_cifar10_cnn.pth",
        map_location=device
    )
)

model.eval()


# Image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# Prediction function
def gradio_predict(image):

    image = image.convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        probabilities = probabilities.cpu().numpy()[0]

    results = {
        classes[i]: float(probabilities[i])
        for i in range(len(classes))
    }

    return results


# Create Gradio interface
interface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=10),
    title="CIFAR-10 Image Classification",
    description="Upload an image to classify it into one of the 10 CIFAR-10 categories."
)


# Launch application
if __name__ == "__main__":
    interface.launch()
