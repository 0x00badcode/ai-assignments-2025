import numpy as np
from typing import Callable, Dict, Type
from abc import ABC, abstractmethod

class Activation(ABC):
    """Abstract base class for activation functions."""
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of activation function."""
        pass
    
    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Backward pass (derivative) of activation function."""
        pass

class ReLU(Activation):
    """Rectified Linear Unit activation function."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

class LeakyReLU(Activation):
    """Leaky ReLU activation function."""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)

class Sigmoid(Activation):
    """Sigmoid activation function."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)

class Tanh(Activation):
    """Hyperbolic tangent activation function."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

class Softmax(Activation):
    """Softmax activation function for multi-class classification."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        # Note: This is a simplified version. In practice, this should be combined with cross-entropy loss
        s = self.forward(x)
        return s * (1 - s)

# Dictionary mapping activation names to their classes
ACTIVATIONS: Dict[str, Type[Activation]] = {
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'softmax': Softmax
}

def get_activation(name: str) -> Activation:
    """Get activation function by name.
    
    Args:
        name: Name of the activation function
        
    Returns:
        Activation function instance
        
    Raises:
        ValueError: If activation function is not found
    """
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation function: {name}")
    return ACTIVATIONS[name]()