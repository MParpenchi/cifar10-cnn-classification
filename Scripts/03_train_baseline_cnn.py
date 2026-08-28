import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# =========================
# Reproducibility
# =========================

np.random.seed(0)
tf.random.set_seed(0)


# =========================
# Project paths
# =========================

PROJECT_DIR = "/Users/maryam/Desktop/Lessons/2th semester/Deep Learning/FDL_CIFAR10_PROJECT"

RESULT_DIR = os.path.join(PROJECT_DIR, "Result")
FIGURES_DIR = os.path.join(PROJECT_DIR, "Figures")
MODELS_DIR = os.path.join(PROJECT_DIR, "Models")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# =========================
# 1. Load prepared dataset
# =========================

prepared_path = os.path.join(RESULT_DIR, "cifar10_prepared.npz")

data = np.load(prepared_path, allow_pickle=True)

x_train = data["x_train"]
y_train = data["y_train"]

x_val = data["x_val"]
y_val = data["y_val"]

x_test = data["x_test"]
y_test = data["y_test"]

class_names = data["class_names"]

num_classes = len(class_names)

print("Dataset loaded successfully.")
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
print("Number of classes:", num_classes)


# =========================
# 2. Convert labels to one-hot encoding
# =========================

# Example:
# label 6 becomes [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print("\nAfter one-hot encoding:")
print("y_train_cat shape:", y_train_cat.shape)
print("y_val_cat shape:", y_val_cat.shape)
print("y_test_cat shape:", y_test_cat.shape)


# =========================
# 3. Define baseline CNN model
# =========================

inputs = keras.Input(shape=(32, 32, 3))

x = inputs

x = keras.layers.Conv2D(32, 3, padding="same")(x)
x = keras.layers.Activation("relu")(x)

x = keras.layers.Conv2D(64, 3, padding="same")(x)
x = keras.layers.Activation("relu")(x)

x = keras.layers.GlobalMaxPooling2D()(x)

outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

net = keras.Model(inputs, outputs)

print("\nModel summary:")
net.summary()


# =========================
# 4. Compile model
# =========================

net.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=["accuracy"]
)


# =========================
# 5. Train model
# =========================

epochs = 15
batch_size = 64

history = net.fit(
    x=x_train,
    y=y_train_cat,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val_cat),
    verbose=1
)


# =========================
# 6. Plot learning curves
# =========================

plt.figure(figsize=(10, 4))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline CNN - Loss")
plt.legend(["train", "validation"])

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Baseline CNN - Accuracy")
plt.legend(["train", "validation"])

plt.tight_layout()

figure_path = os.path.join(FIGURES_DIR, "baseline_cnn_learning_curves.png")
plt.savefig(figure_path, dpi=200)
plt.close()

print("\nLearning curves saved to:")
print(figure_path)


# =========================
# 7. Evaluate on test set
# =========================

test_loss, test_accuracy = net.evaluate(x_test, y_test_cat, verbose=0)

print("\nTest results:")
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


# =========================
# 8. Save model
# =========================

model_path = os.path.join(MODELS_DIR, "baseline_cnn.keras")
net.save(model_path)

print("\nModel saved to:")
print(model_path)


# =========================
# 9. Save results in a text file
# =========================

results_path = os.path.join(RESULT_DIR, "baseline_cnn_results.txt")

with open(results_path, "w") as f:
    f.write("Baseline CNN Results\n")
    f.write("====================\n\n")
    f.write(f"Train samples: {x_train.shape[0]}\n")
    f.write(f"Validation samples: {x_val.shape[0]}\n")
    f.write(f"Test samples: {x_test.shape[0]}\n")
    f.write(f"Number of classes: {num_classes}\n\n")
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Batch size: {batch_size}\n\n")
    f.write(f"Final train loss: {history.history['loss'][-1]}\n")
    f.write(f"Final train accuracy: {history.history['accuracy'][-1]}\n")
    f.write(f"Final validation loss: {history.history['val_loss'][-1]}\n")
    f.write(f"Final validation accuracy: {history.history['val_accuracy'][-1]}\n\n")
    f.write(f"Test loss: {test_loss}\n")
    f.write(f"Test accuracy: {test_accuracy}\n")

print("\nResults saved to:")
print(results_path)

print("\nBaseline CNN training completed successfully.")