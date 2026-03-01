# Simple Multi-layer Perceptron from Scratch
This project implements a Multi-Layer Perceptron (MLP) neural network entirely from scratch using NumPy and Pandas. Unlike high-level libraries (TensorFlow or PyTorch), this implementation manually handles forward propagation, error calculation, and backpropagation via gradient descent to predict a person's BMI index based on physical attributes.
## Overview
The core goal of this project is to demonstrate the fundamental mathematics behind neural networks. It processes a dataset containing height, weight, and gender to categorize BMI levels.
### Key Features
- Manual Backpropagation: Implements the chain rule to calculate gradients and update weights without automated differentiation.
- Custom Activation Function: Utilizes the Sigmoid function for mapping values between 0 and 1.
- Data Normalization: Includes a custom preprocessing pipeline to scale physical units (Height, Weight) and encode categorical data (Gender) for neural network stability.
- Weight Management: Uses randomized weight initialization and an early-stopping mechanism if the error begins to diverge.

---

## Technical Architecture
The network consists of a 3-layer architecture:
1. Input Layer: 3 Neurons (Height, Weight, Gender).
2. Hidden Layer: 3 Neurons with Sigmoid activation.
3. Output Layer: 1 Neuron predicting the scaled BMI Index.
4. This README provides a professional overview of your project, highlighting the technical implementation of building a neural network from the ground up.
### Mathematical Implementation
The project utilizes the following formulas for its learning process:
* **Activation:** 
$$f(x) = \frac{1}{1 + e^{-x}}$$
* **Gradient Descent:** Weights are updated iteratively using the learning rate ($\eta$):
$$w_{new} = w_{old} + \eta \cdot \text{gradient}$$

---

## Project Structure
* `multilayer.py`: The main script containing the model architecture, training loop, and prediction logic.
* `bmi.csv`: The dataset used for training (expects columns: `Height`, `Weight`, `Gender`, `Index`).

---

## 📊 Logic & Normalization
To ensure the model converges, the input data is scaled as follows:
* **Height:** Divided by 1000.
* **Weight:** Divided by 100.
* **Gender:** Mapped to numerical values (**Male: 0.1**, **Female: 0.2**).
* **Target (BMI):** Normalized by dividing by 10 to keep the output within the Sigmoid range.
> **Note:** The `fit` function includes a safeguard that stops training and restores the previous best weights if the square error increases, preventing the model from diverging.
