import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Project paths
# =========================

PROJECT_DIR = "/Users/maryam/Desktop/Lessons/2th semester/Deep Learning/FDL_CIFAR10_PROJECT"

RESULT_DIR = os.path.join(PROJECT_DIR, "Result")
FIGURES_DIR = os.path.join(PROJECT_DIR, "Figures")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# =========================
# 1. Manually collect final results
# =========================

# These values come from the outputs of the training scripts:
# 03_train_baseline_cnn.py
# 04_train_improved_cnn.py
# 06_train_augmented_cnn.py

results = [
    {
        "Model": "Baseline CNN",
        "Main idea": "Simple CNN architecture",
        "Parameters": 20042,
        "Test loss": 1.1902,
        "Test accuracy": 0.5749,
        "Test accuracy (%)": 57.49,
        "Interpretation": "Underfitting; model capacity was too limited."
    },
    {
        "Model": "Improved CNN",
        "Main idea": "Deeper CNN + BatchNorm + Dropout",
        "Parameters": 158506,
        "Test loss": 0.5705,
        "Test accuracy": 0.8109,
        "Test accuracy (%)": 81.09,
        "Interpretation": "Good model; better capacity and regularization."
    },
    {
        "Model": "Augmented CNN",
        "Main idea": "Improved CNN + data augmentation",
        "Parameters": 158506,
        "Test loss": 0.6880,
        "Test accuracy": 0.7669,
        "Test accuracy (%)": 76.69,
        "Interpretation": "Augmentation was probably too strong for 32x32 images."
    },
    {
        "Model": "Improved CNN + LR Scheduler",
        "Main idea": "Improved CNN + ReduceLROnPlateau",
        "Parameters": 158506,
        "Test loss": 0.4925,
        "Test accuracy": 0.8421,
        "Test accuracy (%)": 84.21,
        "Interpretation": "Best model; smoother convergence and highest test accuracy."
    }
]

df = pd.DataFrame(results)

print("\nFinal comparison table:")
print(df)


# =========================
# 2. Save table as CSV
# =========================

csv_path = os.path.join(RESULT_DIR, "model_comparison.csv")
df.to_csv(csv_path, index=False)

print("\nModel comparison CSV saved to:")
print(csv_path)


# =========================
# 3. Save table as TXT
# =========================

txt_path = os.path.join(RESULT_DIR, "model_comparison.txt")

with open(txt_path, "w") as f:
    f.write("Model Comparison Summary\n")
    f.write("========================\n\n")
    f.write(df.to_string(index=False))
    f.write("\n\nConclusion:\n")
    f.write(
        "The improved CNN with learning-rate scheduling achieved the best test accuracy "
        "The Baseline CNN underfit the data due to limited capacity. "
        "The Augmented CNN reduced memorization pressure but achieved lower accuracy, "
        "probably because the selected augmentations were too strong for low-resolution CIFAR-10 images.\n"
    )

print("\nModel comparison TXT saved to:")
print(txt_path)


# =========================
# 4. Save accuracy comparison plot
# =========================

plt.figure(figsize=(8, 5))

plt.bar(df["Model"], df["Test accuracy (%)"])

plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")
plt.title("CIFAR-10 Model Comparison")
plt.ylim(0, 100)

for i, value in enumerate(df["Test accuracy (%)"]):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.tight_layout()

plot_path = os.path.join(FIGURES_DIR, "model_comparison_accuracy.png")
plt.savefig(plot_path, dpi=200)
plt.close()

print("\nAccuracy comparison plot saved to:")
print(plot_path)


# =========================
# 5. Save loss comparison plot
# =========================

plt.figure(figsize=(8, 5))

plt.bar(df["Model"], df["Test loss"])

plt.xlabel("Model")
plt.ylabel("Test Loss")
plt.title("CIFAR-10 Test Loss Comparison")

for i, value in enumerate(df["Test loss"]):
    plt.text(i, value + 0.02, f"{value:.4f}", ha="center")

plt.tight_layout()

loss_plot_path = os.path.join(FIGURES_DIR, "model_comparison_loss.png")
plt.savefig(loss_plot_path, dpi=200)
plt.close()

print("\nLoss comparison plot saved to:")
print(loss_plot_path)


print("\nResult summarization completed successfully.")
