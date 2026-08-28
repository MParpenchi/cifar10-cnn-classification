import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# =========================
# Project paths
# =========================

PROJECT_DIR = "/Users/maryam/Desktop/Lessons/2th semester/Deep Learning/FDL_CIFAR10_PROJECT"

RESULT_DIR = os.path.join(PROJECT_DIR, "Result")
FIGURES_DIR = os.path.join(PROJECT_DIR, "Figures")
MODELS_DIR = os.path.join(PROJECT_DIR, "Models")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# =========================
# 1. Load prepared dataset
# =========================

prepared_path = os.path.join(RESULT_DIR, "cifar10_prepared.npz")
data = np.load(prepared_path, allow_pickle=True)

x_test = data["x_test"]
y_test = data["y_test"]
class_names = data["class_names"]

num_classes = len(class_names)

y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print("Test data loaded.")
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
print("y_test_cat shape:", y_test_cat.shape)
print("Number of classes:", num_classes)


# =========================
# 2. Load final best model
# =========================

model_path = os.path.join(MODELS_DIR, "improved_cnn_lr_scheduler_best.keras")
model = keras.models.load_model(model_path)

print("\nFinal best model loaded from:")
print(model_path)


# =========================
# 3. Evaluate model
# =========================

test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)

print("\nFinal model test results:")
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


# =========================
# 4. Predict classes
# =========================

y_pred_probs = model.predict(x_test, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nPrediction probabilities shape:", y_pred_probs.shape)
print("Predicted labels shape:", y_pred.shape)


# =========================
# 5. Classification report
# =========================

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\nFinal model classification report:")
print(report)

report_path = os.path.join(RESULT_DIR, "final_model_classification_report.txt")

with open(report_path, "w") as f:
    f.write("Final Model Classification Report\n")
    f.write("=================================\n\n")
    f.write("Model: Improved CNN + ReduceLROnPlateau\n\n")
    f.write(f"Test loss: {test_loss}\n")
    f.write(f"Test accuracy: {test_accuracy}\n\n")
    f.write(report)

print("\nFinal classification report saved to:")
print(report_path)


# =========================
# 6. Confusion matrix
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 10))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    xticks_rotation=45,
    cmap="Blues",
    values_format="d"
)

plt.title("Final Model - Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(FIGURES_DIR, "final_model_confusion_matrix.png")
plt.savefig(cm_path, dpi=200)
plt.close()

print("\nFinal confusion matrix saved to:")
print(cm_path)


# =========================
# 7. Correct prediction examples
# =========================

correct_indices = np.where(y_pred == y_test)[0]

plt.figure(figsize=(12, 6))

for i in range(20):
    idx = correct_indices[i]
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_test[idx])
    plt.title(
        f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}",
        fontsize=8
    )
    plt.axis("off")

plt.tight_layout()

correct_path = os.path.join(FIGURES_DIR, "final_model_correct_predictions.png")
plt.savefig(correct_path, dpi=200)
plt.close()

print("\nFinal correct prediction examples saved to:")
print(correct_path)


# =========================
# 8. Wrong prediction examples
# =========================

wrong_indices = np.where(y_pred != y_test)[0]

plt.figure(figsize=(12, 6))

for i in range(20):
    idx = wrong_indices[i]
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_test[idx])
    plt.title(
        f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}",
        fontsize=8
    )
    plt.axis("off")

plt.tight_layout()

wrong_path = os.path.join(FIGURES_DIR, "final_model_wrong_predictions.png")
plt.savefig(wrong_path, dpi=200)
plt.close()

print("\nFinal wrong prediction examples saved to:")
print(wrong_path)


# =========================
# 9. Save predicted labels and probabilities
# =========================

predictions_path = os.path.join(RESULT_DIR, "final_model_predictions.npz")

np.savez_compressed(
    predictions_path,
    y_test=y_test,
    y_pred=y_pred,
    y_pred_probs=y_pred_probs,
    class_names=class_names
)

print("\nFinal predictions saved to:")
print(predictions_path)


print("\nFinal model evaluation completed successfully.")