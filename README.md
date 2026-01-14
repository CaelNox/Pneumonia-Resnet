Medical Image Classification using ResNet-50

This repository contains a deep learning project for classifying chest X-ray images using a convolutional neural network based on the ResNet-50 architecture. The project focuses on building a reliable, modular, and reproducible image classification pipeline using PyTorch, with an emphasis on proper evaluation and experimentation practices.

The work is intended as an applied machine learning project demonstrating model development, training, evaluation, and planned deployment.

--------------------------------------------------------------------

Project Overview

Medical imaging is a key component of modern healthcare, and chest X-rays are among the most commonly used diagnostic tools. This project explores how convolutional neural networks can be trained to learn visual patterns from chest X-ray images and perform automated image classification.

Rather than focusing only on model accuracy, the project emphasizes:
- clean dataset handling
- robust training using cross-validation
- comprehensive evaluation using multiple performance metrics
- code organization suitable for future extensions

--------------------------------------------------------------------

Methodology

The model architecture is based on ResNet-50, pretrained on ImageNet and adapted for medical image classification through transfer learning. The final classification layers are fine-tuned while leveraging the pretrained feature extractor.

Key methodological components include:
- transfer learning using ResNet-50
- image preprocessing and normalization
- stratified k-fold cross-validation to ensure consistent class distribution across folds
- class-weighted loss functions to address dataset imbalance
- separation of training, evaluation, and metric computation logic

The project is implemented entirely in PyTorch.

--------------------------------------------------------------------

Evaluation

Model performance is evaluated using standard classification metrics commonly used in machine learning and medical imaging research.

Metrics computed include:
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-score
- ROC-AUC

Aggregated results across cross-validation folds:

Accuracy: 91.5 percent  
ROC-AUC: 97.5 percent  
Recall (Sensitivity): 91.1 percent  
Specificity: 92.4 percent  
F1-score: 93.9 percent  

These results indicate strong classification performance with balanced sensitivity and specificity.

--------------------------------------------------------------------

Project Structure

Medical_Image_Classification  
src  
dataset.py  
model.py  
train.py  
metrics.py  
utils.py  
config.py  

experiments  
folds  
metrics  

run_train_fold.py  
run_evaluate_fold.py  
README.md  

--------------------------------------------------------------------

Running the Project

Install the required dependencies:

pip install torch torchvision numpy pandas scikit-learn matplotlib

Train the model using cross-validation:

python run_train_fold.py

Evaluate trained models and compute metrics:

python run_evaluate_fold.py

--------------------------------------------------------------------

Future Work

Planned improvements and extensions include:
- a Streamlit-based web interface allowing users to upload a chest X-ray image and receive a classification result from the trained model
- model interpretability techniques such as Grad-CAM for visual explanations
- improved aggregation and visualization of cross-validation metrics

--------------------------------------------------------------------

Dataset

The project uses a publicly available chest X-ray dataset sourced from Kaggle.

--------------------------------------------------------------------

Disclaimer

This project is for academic and educational purposes only. It is not intended for clinical use and should not be relied upon for medical diagnosis or decision-making.

--------------------------------------------------------------------

Author

Aditya Prakash Gupta
