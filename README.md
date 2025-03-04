# Coffee Roasting Neural Network

This repository contains a simple neural network implemented in Python using NumPy and TensorFlow to classify coffee roast levels based on temperature and duration.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Neural Network](#neural-network)
- [Usage](#usage)
- [Visualization](#visualization)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Introduction
This project demonstrates a basic neural network that predicts whether a coffee bean is properly roasted based on its temperature and roasting duration. The implementation includes data preprocessing, normalization, forward propagation through a two-layer neural network, and visualization of the decision boundary.

## Dataset
The dataset contains 200 samples of coffee roasting data with the following features:
- **Temperature** (°F)
- **Roasting Duration** (minutes)
- **Label** (1: Properly roasted, 0: Not properly roasted)

The dataset is loaded using the `load_coffee_data()` function.

## Preprocessing
The data undergoes the following preprocessing steps:
1. **Normalization:** The temperature and duration are normalized using TensorFlow's `Normalization` layer.
2. **Feature Scaling:** Ensures all input features are within a common range for effective training.

## Neural Network
The neural network consists of two layers:
- **First Dense Layer**: Three neurons with a sigmoid activation function.
- **Second Dense Layer**: A single neuron with a sigmoid activation function for binary classification.

### Functions:
- `my_dense(a_in, W, b)`: Implements a fully connected layer.
- `my_sequential(x, W1, b1, W2, b2)`: Implements forward propagation through the two-layer network.
- `my_predict(X, W1, b1, W2, b2)`: Computes predictions for a given input.

## Usage
To test the model, run:
```python
X_tst = np.array([[200, 13.9], [200, 17]])  # Sample test data
X_tstn = norm_l(X_tst)  # Normalize test data
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")
```
Expected output:
```
decisions =
[[1]
 [0]]
```

## Visualization
The decision boundary of the network is visualized using:
```python
netf = lambda x: my_predict(norm_l(x), W1_tmp, b1_tmp, W2_tmp, b2_tmp)
plt_network(X, Y, netf)
```

## Dependencies
Ensure you have the following dependencies installed:
```bash
pip install numpy matplotlib tensorflow
```

## Acknowledgments
This project is based on a machine learning example for binary classification using a simple neural network. Special thanks to `lab_utils_common` and `lab_coffee_utils` for utility functions.

---

Feel free to contribute or raise issues if you find any improvements!


