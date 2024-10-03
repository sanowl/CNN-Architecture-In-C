# Convolutional Neural Network (CNN) Implementation in C

This repository contains an implementation of a simple convolutional neural network (CNN) in C, which is capable of performing forward passes on image data, specifically targeting a structure similar to classic handwritten digit datasets like MNIST.

## Table of Contents
1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Getting Started](#getting-started)
4. [Usage](#usage)
5. [Code Overview](#code-overview)
6. [Contributing](#contributing)
7. [License](#license)

## Features
- Basic Tensor structure for multidimensional data representation.
- Convolutional and pooling layers to extract features from input data.
- Fully connected (FC) layers for classification.
- Activation functions (ReLU, Softmax).
- Cross-entropy loss calculation.
- Multithreaded operations using OpenMP for performance improvements.

## Prerequisites
- GCC or compatible C compiler with support for C11 standard.
- OpenMP library for parallel processing.
- AVX (Advanced Vector Extensions) instructions for SIMD optimizations.

To install required dependencies on Linux (Debian-based distributions):
```sh
sudo apt-get update
sudo apt-get install gcc libomp-dev
```

## Getting Started
To compile the project, run the following command in your terminal:
```sh
gcc -o cnn cnn.c -fopenmp -lm -march=native
```
Where `cnn.c` is the file containing the source code.

To execute the CNN after compiling:
```sh
./cnn
```
This will perform a single forward pass on a sample input tensor and calculate the cross-entropy loss based on a sample target label.

## Usage
1. **Create a Tensor**: Use `create_tensor()` to define multidimensional arrays.
2. **Forward Pass**: The `forward_pass()` function takes the CNN structure and input tensors to compute predictions.
3. **Loss Calculation**: `cross_entropy_loss()` is used to compute the difference between predictions and ground truth.

Modify the input tensor, layer parameters, or target labels as per your requirement in the `main()` function.

## Code Overview
- **Tensor**: Represents multi-dimensional data structures for inputs, weights, etc.
- **ConvLayer**: Represents a convolutional layer with specified input channels, output channels, kernel dimensions, etc.
- **PoolLayer**: Represents a pooling layer, which performs down-sampling.
- **FCLayer**: Represents fully connected layers used for final classification.
- **CNN**: The overall neural network with multiple layers.
- **Operations**:
  - `conv_forward()`: Convolution operation over the input data.
  - `pool_forward()`: Max-pooling operation to reduce feature map dimensions.
  - `fc_forward()`: Fully connected layer computation.
  - `relu()`, `softmax()`, and `cross_entropy_loss()`: Activation and loss functions.

## Contributing
Feel free to open an issue or submit a pull request if you find bugs or want to contribute improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

