# Digit Classification using Artificial Neural Networks (ANN)

![Digit Classification](https://img.shields.io/badge/Digit%20Classification-ANN-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)
![MNIST Dataset](https://img.shields.io/badge/Dataset-MNIST-blue.svg)


A machine learning project for digit classification using Artificial Neural Networks (ANN) and the MNIST dataset. This project demonstrates the power of deep learning in recognizing handwritten digits.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)


## Introduction
The Digit Classification project focuses on recognizing and classifying handwritten digits (0-9) using deep learning techniques. It employs an Artificial Neural Network (ANN) to process and classify the digits, making it a valuable tool for various applications, including optical character recognition (OCR) and digit identification.

## Features
- Input features consist of 28x28 pixel grayscale images of handwritten digits.
- Utilizes a deep learning ANN model for digit classification.
- Provides accuracy metrics and visualizations of the model's performance.

## Dataset

The dataset used in this project is the **MNIST dataset**, a widely recognized and benchmark dataset for digit classification tasks.
### Dataset Overview

- **Number of Samples:** The MNIST dataset consists of a total of 70,000 images.
  - **Training Set:** 60,000 images
  - **Test Set:** 10,000 images

- **Digit Classes:** It is a multi-class classification problem, where each image represents one of the ten digit classes (0-9).

- **Image Dimensions:** All images in the dataset are grayscale and have a fixed size of **28x28 pixels**.

- **Pixel Values:** The pixel values of the images are represented as grayscale values ranging from 0 (black) to 255 (white).


## Model Architecture

The digit classification model in this project is built using the TensorFlow and Keras libraries, specifically employing the `tensorflow.keras.Sequential` API for creating a sequential neural network. Here's an overview of the model architecture:

### Sequential Model

- **Model Type:** The architecture follows a sequential model type, meaning that the neural network is constructed as a linear stack of layers, where each layer has exactly one input tensor and one output tensor.

### Layers

- **Flattening Layer:** prior to the input layer,The data is converted from 2D image into a 1D vector, which is necessary for feeding the data into the subsequent fully connected layers.
  
- **Input Layer:** The initial layer of the model serves as the input layer. It receives the preprocessed 28x28 pixel grayscale images of handwritten digits as input.

- **Fully Connected (Dense) Layers:** The core of the model consists of fully connected (dense) layers. These layers contain multiple neurons (units) that are densely connected to the neurons in the previous layer. Activation functions such as ReLU (Rectified Linear Unit) is applied to introduce non-linearity.

- **Output Layer:** The final layer of the model is an output layer that contains ten neurons, each corresponding to a digit class (0-9). The activation function in the output layer is softmax, which produces probability scores for each class, representing the model's confidence in classifying the input image as a particular digit.

### Compilation

- **Loss Function:** The loss function: sparse categorical cross-entropy is chosen to measure the dissimilarity between the predicted digit class probabilities and the true labels.

- **Optimizer:** An optimizer: Adam  is selected to update the model's weights during training to minimize the loss function.

- **Metrics:** Metrics: accuracy is used to evaluate the model's performance during training and testing.


