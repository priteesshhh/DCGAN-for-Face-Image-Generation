# DCGAN-for-Face-Image-Generation

## Overview
In this assignment, I implemented and trained a Deep Convolutional Generative Adversarial Network (DCGAN) using **PyTorch** to generate realistic facial images based on the **CelebA-HQ 256x256** dataset. The project covers the full deep learning pipeline—from dataset preprocessing to evaluating the generated images using the **Fréchet Inception Distance (FID)** metric.

## Dataset
- **URL**: [Kaggle - CelebA-HQ Resized 256x256](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)

## Major Steps

### 1. Library Installation & Environment Setup
- Installed libraries: `torch`, `torchvision`, `matplotlib`
- Mounted **Google Drive** to access/save datasets and models

### 2. Dataset Handling
- Loaded the **CelebA-HQ 256x256** dataset from Google Drive
- Preprocessing:
  - Resized and center-cropped images
  - Converted images to tensors and normalized to `[-1, 1]`
- Created a custom PyTorch Dataset and DataLoader with batch size **128**

### 3. DCGAN Architecture
- Implemented Generator and Discriminator based on the **DCGAN paper**
- **Generator**:
  - Used `ConvTranspose2d` layers for upsampling
  - Applied `BatchNorm2d` and `ReLU` activations
- **Discriminator**:
  - Used `Conv2d` layers for downsampling
  - Applied `LeakyReLU` and `BatchNorm2d`
- Weight Initialization: Normal distribution

### 4. Training Setup
- **Loss Function**: Binary Cross Entropy (BCE)
- **Optimizer**: Adam (used for both Generator and Discriminator)
- Training Schedule:
  - Initial training: 5 epochs
  - Extended training: 150 epochs total
- Metrics tracked:
  - Generator & Discriminator losses
  - D(x): Discriminator output on real images
  - D(G(z)): Discriminator output on fake images

### 5. Image Generation
- Used a **fixed noise vector** to generate consistent image samples
- Saved generated images in `generated_fakes_1000/`

### 6. FID Score Evaluation
- Installed `pytorch-fid` for evaluation
- Compared generated images with original CelebA dataset
- **FID Score Achieved**: `161.74`

### 7. Saving and Testing
- Saved the **trained Generator** model to Google Drive
- Visualized generated outputs for quality check
- Confirmed sample diversity and dataset integrity

### 8. Results & Observations
- DCGAN effectively learned to generate faces with CelebA-like features
- Generated faces displayed reasonable **structure and diversity**
- FID Score could be improved further through:
  - Longer training
  - Hyperparameter optimization

---

## Final Evaluation
**Fréchet Inception Distance (FID):** _161.74_

## Tools Used
- Google Colab
- PyTorch
- TorchVision
- pytorch-fid
- Matplotlib

## Author
*Priteesh Madhav Reddy Karra*  
*Date: April 2025*
