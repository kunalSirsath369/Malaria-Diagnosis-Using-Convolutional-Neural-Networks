# Malaria Diagnosis Using Convolutional Neural Networks

## Overview
This project aims to develop a convolutional neural network (CNN) model using TensorFlow to diagnose malaria from blood smear images. The model classifies images into two categories: `Parasitized` and `Uninfected`.

## Dataset
The dataset used for this project is the Malaria Cell Images Dataset available from [TensorFlow Datasets](https://www.tensorflow.org/datasets). It contains labeled images of infected and uninfected cells.

## Project Structure

Malaria-Diagnosis/
│
├── data/ # Contains the dataset
│
├── notebooks/ # Jupyter notebooks for exploration and experimentation
│
├── models/ # Trained model files
│
├── src/ # Source code
│ ├── data_preprocessing.py # Data preprocessing and augmentation
│ ├── model.py # Model definition and training script
│ ├── evaluate.py # Model evaluation script
│
├── plots/ # Plot images for model loss and accuracy
│ ├── loss_plot.png
│ ├── accuracy_plot.png
│
├── README.md # Project README
│
└── requirements.txt # Project dependencies
