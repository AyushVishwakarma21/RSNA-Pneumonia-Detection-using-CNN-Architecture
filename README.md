# 🧠 Pneumonia Detection from Chest X-Rays using Deep Learning
## 📍 Project Overview

This project develops a deep learning model to detect pneumonia from chest X-ray images using a ResNet18 architecture.
The work is inspired by the RSNA Pneumonia Detection Challenge and aims to explore how convolutional neural networks (CNNs) can assist in automated radiology diagnostics.

### The model classifies X-rays into two categories:

🫁 Pneumonia

🫀 Normal

### 🧩 Key Features

✅ Built a ResNet18 CNN model from scratch using PyTorch

✅ Achieved ~84% validation accuracy on the RSNA dataset

✅ Implemented data augmentation and class imbalance handling

✅ Visualized model explainability using Grad-CAM

✅ Includes clean, modular code and easy-to-run pipeline


### 🧠 Dataset

Dataset: RSNA Pneumonia Detection Challenge (Kaggle)

Split	Class	Count
Train	Pneumonia	57,199
Train	Normal	10,664
Test	Balanced subset	8,482

### 🧩 Preprocessing

Resized images to 224×224

Normalized intensity values

Applied random flips, rotations, and brightness/contrast augmentations

### ⚙️ Model Architecture

Base model: ResNet18
Input channels: 1 (grayscale)
Output: 2 classes → Pneumonia / Normal
Loss Function: Weighted Cross Entropy
Optimizer: Adam (lr=1e-4)
Scheduler: Cosine Annealing
Epochs: 20
Batch size: 32

ResNet18(
  (conv1): Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
  (layer1–4): Residual Blocks
  (fc): Linear(512, 2)
)

### 📈 Results
Metric	Validation
Accuracy	83.9%
Precision	80.3%
Recall	84.7%
F1-score	82.4%
AUC	0.89

### 🩻 Grad-CAM Visualization

Visual explanations highlight pneumonia-affected lung regions:
Heatmaps focus on opacities and consolidation — matching clinical patterns of pneumonia.


### 🧪 Experiments Conducted
Experiment	Description	Result
Baseline	ResNet18 (scratch)	84% accuracy
Augmented	Added brightness + rotation	+2% gain
Weighted Loss	To fix class imbalance	Slight recall boost
Grad-CAM	Explainability	Heatmaps align with pneumonia zones

#### 📚 Future Work:
Try DenseNet121 and EfficientNetV2 for feature richness

Apply focal loss or class-balanced sampling

Use 5-fold cross-validation for more robust results

Add confidence calibration and uncertainty estimation

Build a Streamlit web demo for model explainability



### ⚠️ Disclaimer

This project is for research and educational purposes only and not intended for clinical diagnosis.
Predictions from this model should not be used for medical decision-making without expert validation.

### 💬 Acknowledgements

RSNA Pneumonia Detection Challenge

PyTorch

Grad-CAM paper (Selvaraju et al., ICCV 2017)

