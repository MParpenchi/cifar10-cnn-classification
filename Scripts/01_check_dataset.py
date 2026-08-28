import os
import scipy.io as sio
import numpy as np

# =========================
# Project paths
# =========================

PROJECT_DIR = "/Users/maryam/Desktop/Lessons/2th semester/Deep Learning/FDL_CIFAR10_PROJECT"
DATA_DIR = os.path.join(PROJECT_DIR, "Data")

# =========================
# File names
# =========================

batch_files = [
    "data_batch_1.mat",
    "data_batch_2.mat",
    "data_batch_3.mat",
    "data_batch_4.mat",
    "data_batch_5.mat",
]

test_file = "test_batch.mat"
meta_file = "batches.meta.mat"

# =========================
# 1. Check that all files exist
# =========================

print("Checking files...\n")

for filename in batch_files + [test_file, meta_file]:
    path = os.path.join(DATA_DIR, filename)
    print(filename, "exists:", os.path.exists(path))

# =========================
# 2. Load one training batch
# =========================

batch1_path = os.path.join(DATA_DIR, "data_batch_1.mat")
batch1 = sio.loadmat(batch1_path)

print("\nKeys inside data_batch_1.mat:")
print(batch1.keys())

x_batch1 = batch1["data"]
y_batch1 = batch1["labels"]

print("\nShape of x_batch1:", x_batch1.shape)
print("Shape of y_batch1:", y_batch1.shape)
print("Data type of x_batch1:", x_batch1.dtype)
print("Data type of y_batch1:", y_batch1.dtype)

# =========================
# 3. Load class names
# =========================

meta_path = os.path.join(DATA_DIR, meta_file)
meta = sio.loadmat(meta_path)

print("\nKeys inside batches.meta.mat:")
print(meta.keys())

label_names_raw = meta["label_names"]

label_names = []
for i in range(label_names_raw.shape[0]):
    label_names.append(str(label_names_raw[i, 0][0]))

print("\nClass names:")
for i, name in enumerate(label_names):
    print(i, name)

# =========================
# 4. Check one image
# =========================

one_image_flat = x_batch1[0]  # shape: (3072,)

print("\nOne flat image shape:", one_image_flat.shape)

# CIFAR-10 MATLAB format:
# first 1024 values = red channel
# next 1024 values = green channel
# last 1024 values = blue channel
one_image = one_image_flat.reshape(3, 32, 32).transpose(1, 2, 0)

print("One reshaped image shape:", one_image.shape)
print("Min pixel value:", one_image.min())
print("Max pixel value:", one_image.max())

first_label = y_batch1[0, 0]
print("First label:", first_label)
print("First label name:", label_names[first_label])