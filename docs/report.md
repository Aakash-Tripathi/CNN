# Use of linear and matrix algebra in Convolutional neural networks

## Abstract
In this report, we will discuss the use of linear and matrix algebra in a class of neural networks called Convolutional Neural Networks (CNNs). We will also build a convolutional neural network from scratch using only numpy, a python library for scientific computing, to understand the inner workings of the network. Finally, we will show a typical implementation of a convolutional neural network using a deep learning framework like PyTorch.

## 1. Introduction
Convolutional neural networks (CNNs) are a class of deep neural networks. Deep neural networks are a class of neural networks that have multiple hidden layers. CNNs are used in image classification, object detection, image segmentation, and many other applications. CNNs are a special type of neural networks that are used to process images. CNNs are used to extract features from images. These features are then used to classify the image. CNNs are made up of convolutional layers, pooling layers, and fully connected layers. Convolutional layers are used to extract features from the image. Pooling layers are used to reduce the dimensionality of the image. Fully connected layers are used to classify the image.

## 2. Background

This section outlines the necessary background knowledge needed for the convolutional neural network. 

### 2.1 Neural Networks

Neural networks are a class of machine learning algorithms that are inspired by the human brain. Neural networks are made up of neurons. Each neuron is connected to other neurons. Each neuron has an input, an activation function, and an output. 

The input of a neuron is the sum of the products of the weights and the inputs of the neurons connected to it.

The activation function is applied to the input of the neuron. 

The output of the neuron is the result of the activation function applied to the input of the neuron. 

The weights of the neurons are updated using gradient descent. 

The weights of the neurons are updated in such a way that the output of the neuron is closer to the desired output. The desired output is the output that we want the neural network to produce. The desired output is also called the target output. The desired output is compared to the output of the neuron to calculate the error. 

The error is then used to update the weights of the neurons. The weights of the neurons are updated in such a way that the error is reduced. The process of updating the weights of the neurons is called backpropagation. 

The process of updating the weights of the neurons is repeated until the error is reduced to an acceptable level. The process of updating the weights of the neurons is called training. 

The process of updating the weights of the neurons is repeated for each training example. The process of updating the weights of the neurons is repeated for each epoch. 

An epoch is a single pass through the entire training set. The process of updating the weights of the neurons is repeated for a fixed number of epochs. The process of updating the weights of the neurons is repeated until the error is reduced to an acceptable level. 

### 2.2 Gradient Descent

Gradient descent is an optimization algorithm used to find the minimum of a function. Gradient descent is used to update the weights of the neurons in a neural network. Gradient descent can be defined as follows: 

\begin{equation}
\theta_{i+1} = \theta_{i} - \alpha \frac{\partial f}{\partial \theta}
\end{equation}

where $\theta$ is the parameter to be optimized, $\alpha$ is the learning rate, and $f$ is the function to be optimized.

### 2.3 Forward Propagation
### 2.4 Backpropagation

## 3. Convolutional Neural Networks

## 4. Linear and Matrix Algebra in Convolutional Neural Networks

## 5. Implementation of a Convolutional Neural Network
### 5.1 From scratch implementation using Numpy
### 5.2 Deep learning framework implementation