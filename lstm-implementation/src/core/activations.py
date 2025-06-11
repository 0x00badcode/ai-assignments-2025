"""
Activation functions for LSTM implementation.
Includes forward pass and derivative calculations.
"""
import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    # Clip to prevent overflow
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Hyperbolic tangent activation function."""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh function."""
    return 1 - np.tanh(x) ** 2

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU function."""
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of Leaky ReLU function."""
    return np.where(x > 0, 1.0, alpha)

def elu(x, alpha=1.0):
    """Exponential Linear Unit (ELU) activation function."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    """Derivative of ELU function."""
    return np.where(x > 0, 1.0, elu(x, alpha) + alpha)

def softmax(x, axis=-1):
    """Softmax activation function with numerical stability."""
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def softmax_derivative(x):
    """Derivative of softmax function."""
    s = softmax(x)
    return s * (1 - s)

# Dictionary mapping activation names to functions
ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'relu': (relu, relu_derivative),
    'leaky_relu': (leaky_relu, leaky_relu_derivative),
    'elu': (elu, elu_derivative),
    'softmax': (softmax, softmax_derivative)
}

def get_activation(name):
    """Get activation function and its derivative by name."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name] 