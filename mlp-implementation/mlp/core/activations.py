"""
Activation Functions for MLP Implementation
Includes ReLU, Sigmoid, Tanh, Softmax, and Linear activations
"""

import numpy as np
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    """Base class for activation functions"""
    
    @abstractmethod
    def forward(self, x):
        """Forward pass"""
        pass
    
    @abstractmethod
    def backward(self, x):
        """Backward pass - derivative"""
        pass


class ReLU(ActivationFunction):
    """Rectified Linear Unit activation function"""
    
    def forward(self, x):
        """ReLU forward: max(0, x)"""
        return np.maximum(0, x)
    
    def backward(self, x):
        """ReLU derivative: 1 if x > 0, else 0"""
        return (x > 0).astype(float)


class Sigmoid(ActivationFunction):
    """Sigmoid activation function"""
    
    def forward(self, x):
        """Sigmoid forward: 1 / (1 + exp(-x))"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        """Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))"""
        s = self.forward(x)
        return s * (1 - s)


class Tanh(ActivationFunction):
    """Hyperbolic tangent activation function"""
    
    def forward(self, x):
        """Tanh forward: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
        return np.tanh(x)
    
    def backward(self, x):
        """Tanh derivative: 1 - tanh²(x)"""
        return 1 - np.power(self.forward(x), 2)


class Softmax(ActivationFunction):
    """Softmax activation function (for multi-class classification)"""
    
    def forward(self, x):
        """Softmax forward: exp(x_i) / sum(exp(x_j))"""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, x):
        """Softmax derivative (computed in loss function for efficiency)"""
        s = self.forward(x)
        return s * (1 - s)


class Linear(ActivationFunction):
    """Linear activation function (identity)"""
    
    def forward(self, x):
        """Linear forward: x"""
        return x
    
    def backward(self, x):
        """Linear derivative: 1"""
        return np.ones_like(x)


class LeakyReLU(ActivationFunction):
    """Leaky ReLU activation function"""
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        """Leaky ReLU forward: max(alpha*x, x)"""
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x):
        """Leaky ReLU derivative: 1 if x > 0, else alpha"""
        return np.where(x > 0, 1.0, self.alpha)


class ELU(ActivationFunction):
    """Exponential Linear Unit activation function"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def forward(self, x):
        """ELU forward: x if x > 0, else alpha*(exp(x) - 1)"""
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def backward(self, x):
        """ELU derivative: 1 if x > 0, else alpha*exp(x)"""
        return np.where(x > 0, 1.0, self.alpha * np.exp(x))


# Factory function for easy activation creation
def get_activation(name, **kwargs):
    """Get activation function by name"""
    activations = {
        'relu': ReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'softmax': Softmax,
        'linear': Linear,
        'leaky_relu': LeakyReLU,
        'elu': ELU
    }
    
    if name.lower() in activations:
        return activations[name.lower()](**kwargs)
    else:
        raise ValueError(f"Unknown activation function: {name}") 