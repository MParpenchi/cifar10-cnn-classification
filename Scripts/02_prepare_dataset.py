import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# =========================
# Project paths
# =========================

PROJECT_DIR = "/Users/maryam/Desktop/Lessons/2th semester/Deep Learning/FDL_CIFAR10_PROJECT"

DATA_DIR = os.path.join(PROJECT_DIR, "Data")
FIGURES_DIR = os.path.join(PROJECT_DIR, "Figures")
RESULT_DIR = os.path.join(PROJECT_DIR, "Result")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


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
# Helper function
# =========================

def load_cifar_batch(file_path):
    """
    Load one CIFAR-10 MATLAB batch.

    Original CIFAR-10 MATLAB data shape:
        data: (10000, 3072)

    Each image is stored as:
        first 1024 values  = red channel
        next 1024 values   = green channel
        last 1024 values   = blue channel

    We convert it to:
        images: (10000, 32, 32, 3)
        labels: (10000,)
    """

    batch = sio.loadmat(file_path)

    x = batch["data"]
    y = batch["labels"].reshape(-1)

    # Convert from flat vector to image format
    x = x.reshape(-1, 3, 32, 32)
    x = x.transpose(0, 2, 3, 1)

    return x, y


# =========================
# 1. Load class names
# =========================

meta = sio.loadmat(os.path.join(DATA_DIR, meta_file))
label_names_raw = meta["label_names"]

class_names = []
for i in range(label_names_raw.shape[0]):
    class_names.append(str(label_names_raw[i, 0][0]))

print("Class names:")
for i, name in enumerate(class_names):
    print(i, name)


# =========================
# 2. Load all training batches
# =========================

x_train_list = []
y_train_list = []

for filename in batch_files:
    file_path = os.path.join(DATA_DIR, filename)

    x_batch, y_batch = load_cifar_batch(file_path)

    x_train_list.append(x_batch)
    y_train_list.append(y_batch)

    print(filename, "loaded:", x_batch.shape, y_batch.shape)


x_all_train = np.concatenate(x_train_list, axis=0)
y_all_train = np.concatenate(y_train_list, axis=0)

print("\nAll training data:")
print("x_all_train shape:", x_all_train.shape)
print("y_all_train shape:", y_all_train.shape)


# =========================
# 3. Load test batch
# =========================

x_test, y_test = load_cifar_batch(os.path.join(DATA_DIR, test_file))

print("\nTest data:")
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# =========================
# 4. Normalize pixel values
# =========================

# Before normalization: pixel values are in [0, 255]
# After normalization: pixel values are in [0, 1]

x_all_train = x_all_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("\nAfter normalization:")
print("x_all_train min:", x_all_train.min())
print("x_all_train max:", x_all_train.max())
print("x_test min:", x_test.min())
print("x_test max:", x_test.max())


# =========================
# 5. Create training and validation sets
# =========================

# We keep the official test set untouched.
# From the 50000 training images, we take 20% as validation.

x_train, x_val, y_train, y_val = train_test_split(
    x_all_train,
    y_all_train,
    test_size=0.20,
    random_state=0,
    stratify=y_all_train
)

print("\nFinal split:")
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# =========================
# 6. Check class distribution
# =========================

print("\nClass distribution in training set:")
for class_id, class_name in enumerate(class_names):
    count = np.sum(y_train == class_id)
    print(class_id, class_name, count)

print("\nClass distribution in validation set:")
for class_id, class_name in enumerate(class_names):
    count = np.sum(y_val == class_id)
    print(class_id, class_name, count)

print("\nClass distribution in test set:")
for class_id, class_name in enumerate(class_names):
    count = np.sum(y_test == class_id)
    print(class_id, class_name, count)


# =========================
# 7. Save a sample image grid
# =========================

plt.figure(figsize=(10, 5))

for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i]], fontsize=8)
    plt.axis("off")

plt.tight_layout()

figure_path = os.path.join(FIGURES_DIR, "sample_cifar10_images.png")
plt.savefig(figure_path, dpi=200)
plt.close()

print("\nSample image grid saved to:")
print(figure_path)


# =========================
# 8. Save prepared dataset
# =========================

prepared_path = os.path.join(RESULT_DIR, "cifar10_prepared.npz")

np.savez_compressed(
    prepared_path,
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test,
    class_names=np.array(class_names)
)

print("\nPrepared dataset saved to:")
print(prepared_path)

print("\nDataset preparation completed successfully.")