# Malaria Diagnosis Using Convolutional Neural Networks

## Overview
This project aims to develop a convolutional neural network (CNN) model using TensorFlow to diagnose malaria from blood smear images. The model classifies images into two categories: `Parasitized` and `Uninfected`.

## Dataset
The dataset used for this project is the Malaria Cell Images Dataset available from [TensorFlow Datasets](https://www.tensorflow.org/datasets). It contains labeled images of infected and uninfected cells.

## Model Architecture
The model is a modified LeNet-5 architecture with added data augmentation and regularization to improve performance and prevent overfitting. The architecture includes the following layers:

Input Layer: Input size of 224x224x3
Data Augmentation: Random flip and random rotation
Convolutional Layers: Two Conv2D layers with ReLU activation and BatchNormalization
Pooling Layers: Two MaxPool2D layers
Dropout Layers: To prevent overfitting
Fully Connected Layers: Dense layers with ReLU activation and BatchNormalization
Output Layer: Dense layer with sigmoid activation for binary classification

## Training and Validation
The model is trained for 20 epochs with an initial learning rate of 0.01, which is reduced on plateau. Early stopping is used to halt training when the validation loss stops improving. The final model is evaluated on a separate test set.

## Training and Validation Loss Plot

## Training and Validation Accuracy Plot

## Actual and Predicated Result

## References
TensorFlow Documentation: https://www.tensorflow.org/
TensorFlow Datasets: https://www.tensorflow.org/datasets/catalog/malaria
