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
# 2. One-hot encoding
# =========================

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print("\nAfter one-hot encoding:")
print("y_train_cat shape:", y_train_cat.shape)
print("y_val_cat shape:", y_val_cat.shape)
print("y_test_cat shape:", y_test_cat.shape)


# =========================
# 3. Define improved CNN model
# =========================

inputs = keras.Input(shape=(32, 32, 3))

x = inputs

# Block 1
x = keras.layers.Conv2D(32, 3, padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation("relu")(x)

x = keras.layers.Conv2D(32, 3, padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation("relu")(x)

x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Dropout(0.25)(x)

# Block 2
x = keras.layers.Conv2D(64, 3, padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation("relu")(x)

x = keras.layers.Conv2D(64, 3, padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation("relu")(x)

x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Dropout(0.25)(x)

# Block 3
x = keras.layers.Conv2D(128, 3, padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation("relu")(x)

x = keras.layers.GlobalAveragePooling2D()(x)

# Classifier
x = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.Dropout(0.40)(x)

outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

net = keras.Model(inputs, outputs)

print("\nModel summary:")
net.summary()


# =========================
# 4. Compile model
# =========================

net.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)


# =========================
# 5. Callbacks
# =========================

checkpoint_path = os.path.join(MODELS_DIR, "improved_cnn_best.keras")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]


# =========================
# 6. Train model
# =========================

epochs = 30
batch_size = 64

history = net.fit(
    x=x_train,
    y=y_train_cat,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val_cat),
    callbacks=callbacks,
    verbose=1
)


# =========================
# 7. Plot learning curves
# =========================

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Improved CNN - Loss")
plt.legend(["train", "validation"])

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Improved CNN - Accuracy")
plt.legend(["train", "validation"])

plt.tight_layout()

figure_path = os.path.join(FIGURES_DIR, "improved_cnn_learning_curves.png")
plt.savefig(figure_path, dpi=200)
plt.close()

print("\nLearning curves saved to:")
print(figure_path)


# =========================
# 8. Evaluate on test set
# =========================

test_loss, test_accuracy = net.evaluate(x_test, y_test_cat, verbose=0)

print("\nTest results:")
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


# =========================
# 9. Save final model
# =========================

final_model_path = os.path.join(MODELS_DIR, "improved_cnn_final.keras")
net.save(final_model_path)

print("\nFinal model saved to:")
print(final_model_path)

print("\nBest model saved to:")
print(checkpoint_path)


# =========================
# 10. Save results in a text file
# =========================

results_path = os.path.join(RESULT_DIR, "improved_cnn_results.txt")

with open(results_path, "w") as f:
    f.write("Improved CNN Results\n")
    f.write("====================\n\n")
    f.write(f"Train samples: {x_train.shape[0]}\n")
    f.write(f"Validation samples: {x_val.shape[0]}\n")
    f.write(f"Test samples: {x_test.shape[0]}\n")
    f.write(f"Number of classes: {num_classes}\n\n")
    f.write(f"Epochs requested: {epochs}\n")
    f.write(f"Epochs actually trained: {len(history.history['loss'])}\n")
    f.write(f"Batch size: {batch_size}\n\n")
    f.write(f"Final train loss: {history.history['loss'][-1]}\n")
    f.write(f"Final train accuracy: {history.history['accuracy'][-1]}\n")
    f.write(f"Final validation loss: {history.history['val_loss'][-1]}\n")
    f.write(f"Final validation accuracy: {history.history['val_accuracy'][-1]}\n\n")
    f.write(f"Test loss: {test_loss}\n")
    f.write(f"Test accuracy: {test_accuracy}\n")

print("\nResults saved to:")
print(results_path)

print("\nImproved CNN training completed successfully.")