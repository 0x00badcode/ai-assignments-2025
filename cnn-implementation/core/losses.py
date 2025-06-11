import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Type

class Loss(ABC):
    """Abstract base class for loss functions."""
    
    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the loss value.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Loss value
        """
        pass
    
    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient of the loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Loss gradient
        """
        pass

class MSE(Loss):
    """Mean Squared Error loss for regression tasks."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.square(y_pred - y_true))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_pred.size

class CrossEntropy(Loss):
    """Cross-Entropy loss for classification tasks."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Add small epsilon for numerical stability
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Add small epsilon for numerical stability
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true / y_pred) / y_pred.shape[0]

class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy loss for binary classification."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Add small epsilon for numerical stability
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Add small epsilon for numerical stability
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_pred.shape[0]

class HuberLoss(Loss):
    """Huber loss for regression tasks, less sensitive to outliers than MSE."""
    
    def __init__(self, delta: float = 1.0):
        """Initialize Huber loss.
        
        Args:
            delta: Threshold for switching between quadratic and linear loss
        """
        self.delta = delta
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        error = y_pred - y_true
        is_small_error = np.abs(error) <= self.delta
        squared_loss = 0.5 * np.square(error)
        linear_loss = self.delta * np.abs(error) - 0.5 * np.square(self.delta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        error = y_pred - y_true
        is_small_error = np.abs(error) <= self.delta
        return np.where(is_small_error, error, self.delta * np.sign(error)) / y_pred.shape[0]

# Dictionary mapping loss names to their classes
LOSSES: Dict[str, Type[Loss]] = {
    'mse': MSE,
    'cross_entropy': CrossEntropy,
    'binary_cross_entropy': BinaryCrossEntropy,
    'huber': HuberLoss
}

def get_loss(name: str) -> Loss:
    """Get loss function by name.
    
    Args:
        name: Name of the loss function
        
    Returns:
        Loss function instance
        
    Raises:
        ValueError: If loss function is not found
    """
    if name not in LOSSES:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSSES[name]()