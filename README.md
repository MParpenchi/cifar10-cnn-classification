# CIFAR-10 Image Classification with Convolutional Neural Networks

Deep Learning project developed for the **Foundations of Deep Learning** course at the **University of Milano-Bicocca**.

## Project Overview

This project focuses on multi-class image classification using Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset.

The goal was to develop and compare different CNN architectures and training strategies, starting from a simple baseline model and gradually improving its performance.

## Dataset

The CIFAR-10 dataset contains **60,000 RGB images** belonging to 10 balanced classes.

- 40,000 training images
- 10,000 validation images
- 10,000 test images
- Image size: 32×32×3

## Models & Results

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| Baseline CNN | 57.49% | 1.1902 |
| Improved CNN | 81.09% | 0.5705 |
| CNN + Data Augmentation | 76.69% | 0.6880 |
| Improved CNN + LR Scheduler | **84.21%** | **0.4925** |

### Final Model

The best-performing model was the **Improved CNN with ReduceLROnPlateau**.

- **Test Accuracy:** 84.21%
- **Macro F1-score:** 84.14%
- **Best Validation Accuracy:** 84.17%
- **Test Loss:** 0.4925

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Convolutional Neural Networks
- Computer Vision

## Techniques

- Image preprocessing and normalization
- CNN architecture design
- Batch Normalization
- MaxPooling
- Dropout
- Global Average Pooling
- Data Augmentation
- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau
- Confusion Matrix Analysis

## Repository Structure

```text
├── Figures/
├── Models/
├── Presentation/
├── Result/
├── Scripts/
└── README.md
