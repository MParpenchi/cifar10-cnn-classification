CIFAR-10 Image Classification with Convolutional Neural Networks
Foundations of Deep Learning Project

Group members:
Mahsa Rajabi Nejad
Maryamsadat Parpenchi

Project description:
This project focuses on CIFAR-10 image classification using Convolutional Neural Networks.
The goal is to classify 32x32 RGB images into one of 10 classes.

Dataset:
CIFAR-10
Total images: 60,000
Training images: 50,000
Test images: 10,000
Classes: 10
Image size: 32x32x3


Dataset source:
CIFAR-10 official website:
https://www.cs.toronto.edu/~kriz/cifar.html

Dataset format used in this project:
CIFAR-10 MATLAB version (.mat files)

Note:
The dataset files are not included in the submission because of their size.
They can be downloaded from the official CIFAR-10 website

Final selected model:
Improved CNN + ReduceLROnPlateau

Final model file:
Models/improved_cnn_lr_scheduler_best.keras

Final test results:
Test accuracy: 84.21%
Test loss: 0.4925
Macro F1-score: 84.14%

Main scripts:
01_check_dataset.py
- Checks that all CIFAR-10 .mat files exist and can be loaded correctly.

02_prepare_dataset.py
- Loads all CIFAR-10 batches.
- Reshapes images from 3072 vectors to 32x32x3.
- Normalizes pixel values from [0, 255] to [0, 1].
- Splits the training data into train and validation sets.
- Saves the prepared dataset.

03_train_baseline_cnn.py
- Trains a simple baseline CNN.
- Test accuracy: 57.49%

04_train_improved_cnn.py
- Trains a deeper CNN with Batch Normalization, MaxPooling, Dropout, and Adam optimizer.
- Test accuracy: 81.09%

05_evaluate_improved_cnn.py
- Evaluates the improved CNN.
- Saves classification report, confusion matrix, and correct/wrong prediction examples.

06_train_augmented_cnn.py
- Tests data augmentation.
- Test accuracy: 76.69%
- The augmentation policy was probably too strong for 32x32 images.

07_summarize_results.py
- Summarizes model results.
- Saves model comparison table and accuracy/loss plots.

08_train_improved_cnn_lr_scheduler.py
- Trains the improved CNN with ReduceLROnPlateau.
- This is the final selected model.
- Test accuracy: 84.21%

09_evaluate_final_model.py
- Evaluates the final model.
- Saves final classification report, final confusion matrix, and correct/wrong prediction examples.

Important output files:
Figures/model_comparison_accuracy.png
Figures/model_comparison_loss.png
Figures/final_model_confusion_matrix.png
Figures/final_model_correct_predictions.png
Figures/final_model_wrong_predictions.png

Result/final_model_classification_report.txt
Result/model_comparison.txt

How to run the project:
Run the scripts in numerical order from the project root folder.

Example:
python Scripts/01_check_dataset.py
python Scripts/02_prepare_dataset.py
python Scripts/03_train_baseline_cnn.py
python Scripts/04_train_improved_cnn.py
python Scripts/06_train_augmented_cnn.py
python Scripts/08_train_improved_cnn_lr_scheduler.py
python Scripts/09_evaluate_final_model.py

Note:
The final model is already trained and saved in:
Models/improved_cnn_lr_scheduler_best.keras