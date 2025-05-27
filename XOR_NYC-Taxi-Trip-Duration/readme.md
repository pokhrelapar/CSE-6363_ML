# Assignment 2 — Neural Network Library and Applications

## Overview

This assignment consists of two parts:

1. **Building a Neural Network Library from Scratch**
2. **Applying the Library to Solve the XOR Problem and Predict NYC Taxi Trip Durations**

---

## 🧠 Part 1: Neural Network Library

Construct a neural network library using Python and NumPy, designed with modularity and extensibility in mind.

### ✅ Features

- **Layer-based Architecture:** All layers inherit from a base `Layer` class that defines the `forward` and `backward` methods.
- **Implemented Layers:**
  - `Linear`: Fully connected layer
  - `Sigmoid`: Logistic activation function
  - `ReLU`: Rectified Linear Unit activation
- **Loss Function:**
  - `BinaryCrossEntropy`: For binary classification tasks
- **Sequential Container:** Easily stack layers using the `Sequential` class
- **Model Persistence:** Save/load model weights via file

### 🧪 XOR Problem

To test the library, construct a 1-hidden-layer neural network with 2 hidden units to solve the XOR classification problem.

- Train the network using:
  - **Sigmoid activation**
  - **Tanh activation**
- Compare training performance and convergence
- Save final weights as: `XOR_solved.w`

---

## 🚕 Part 2: NYC Taxi Trip Duration Prediction

Using the neural network library to build regression models for predicting taxi trip durations.
