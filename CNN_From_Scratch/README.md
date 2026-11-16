
# 🖼️ CIFAR-10 Image Classification Using CNN From Scratch

### Deep Learning | Computer Vision | PyTorch

<p align="center">
  <strong>Building a Convolutional Neural Network from scratch to classify real-world images into 10 different categories.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Computer%20Vision-Deep%20Learning-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Gradio-Deployment-orange?style=for-the-badge">
</p>

---

## 📌 Project Overview

This project focuses on building a **Convolutional Neural Network (CNN) from scratch** using PyTorch for image classification on the CIFAR-10 dataset.

The model learns visual features through multiple convolutional layers and classifies images into 10 different categories.

The project covers the complete deep learning workflow, from dataset preparation and model development to training, evaluation, and deployment.

## 📂 Dataset

**CIFAR-10 Dataset**

- 50,000 training images
- 10,000 testing images
- Image size: 32 × 32 pixels
- Image channels: 3 (RGB)
- Number of classes: 10

### 🎯 Classes

✈️ Airplane | 🚗 Automobile | 🐦 Bird | 🐱 Cat | 🦌 Deer

🐶 Dog | 🐸 Frog | 🐴 Horse | 🚢 Ship | 🚚 Truck

## ⚙️ Model Architecture

A custom CNN architecture was implemented using PyTorch.

**Architecture Highlights:**

- 4 Convolutional Blocks
- Convolutional Layers with 3 × 3 filters
- Batch Normalization
- ReLU Activation
- Max Pooling
- Dropout for Regularization
- Adaptive Average Pooling
- Fully Connected Layers
- Softmax-based class probabilities during prediction

## 🔄 Project Workflow

1. **Data Preparation**
   - Load CIFAR-10 dataset
   - Explore images and labels
   - Apply data augmentation
   - Split training data into training and validation sets

2. **Model Development**
   - Build CNN architecture from scratch
   - Define loss function and optimizer
   - Configure learning rate scheduler

3. **Model Training**
   - Train the CNN using PyTorch
   - Monitor training and validation performance
   - Implement early stopping
   - Save the best model checkpoint

4. **Model Evaluation**
   - Evaluate on unseen test data
   - Generate confusion matrix
   - Generate classification report

5. **Deployment**
   - Build an interactive Gradio interface
   - Upload an image and get predicted class probabilities

## 📊 Model Performance

| Metric | Accuracy |
|---|---:|
| Best Validation Accuracy | 85.36% |
| Test Accuracy | 84.60% |

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Gradio

## 📁 Project Structure

```text
CNN_From_Scratch/
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── deployment.py
│
├── models/
├── plots/
├── data/
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 How to Run

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CNN_From_Scratch
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Train the model**

```bash
python -m src.train
```

**4. Evaluate the model**

```bash
python -m src.evaluate
```

**5. Launch the Gradio application**

```bash
python -m src.deployment
```

## 🔮 Future Improvements

- Experiment with deeper CNN architectures
- Improve classification accuracy through hyperparameter tuning
- Compare CNN from scratch with transfer learning
- Experiment with additional data augmentation techniques

---

<p align="center">
  <strong>Learning Deep Learning by Building Models from Scratch 🚀</strong>
</p>

