"""
Loss Functions for MLP Implementation
Includes MSE, Cross-Entropy, and Binary Cross-Entropy losses
"""

import numpy as np
from abc import ABC, abstractmethod


class LossFunction(ABC):
    """Base class for loss functions"""
    
    @abstractmethod
    def forward(self, y_true, y_pred):
        """Compute loss"""
        pass
    
    @abstractmethod
    def backward(self, y_true, y_pred):
        """Compute gradient of loss w.r.t predictions"""
        pass


class MeanSquaredError(LossFunction):
    """Mean Squared Error loss (for regression)"""
    
    def forward(self, y_true, y_pred):
        """MSE forward: (1/2n) * sum((y_true - y_pred)²)"""
        return np.mean(0.5 * np.power(y_true - y_pred, 2))
    
    def backward(self, y_true, y_pred):
        """MSE gradient: -(y_true - y_pred) / n"""
        return -(y_true - y_pred) / y_true.shape[0]


class MeanAbsoluteError(LossFunction):
    """Mean Absolute Error loss (for regression)"""
    
    def forward(self, y_true, y_pred):
        """MAE forward: (1/n) * sum(|y_true - y_pred|)"""
        return np.mean(np.abs(y_true - y_pred))
    
    def backward(self, y_true, y_pred):
        """MAE gradient: -sign(y_true - y_pred) / n"""
        return -np.sign(y_true - y_pred) / y_true.shape[0]


class CrossEntropy(LossFunction):
    """Cross-Entropy loss (for multi-class classification)"""
    
    def forward(self, y_true, y_pred):
        """Cross-entropy forward: -sum(y_true * log(y_pred))"""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        if y_true.ndim == 1:
            # Convert to one-hot if needed
            num_classes = y_pred.shape[1]
            y_true_onehot = np.zeros((y_true.shape[0], num_classes))
            y_true_onehot[np.arange(y_true.shape[0]), y_true.astype(int)] = 1
            y_true = y_true_onehot
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def backward(self, y_true, y_pred):
        """Cross-entropy gradient: -(y_true / y_pred) / n"""
        # Clip predictions to prevent division by 0
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        if y_true.ndim == 1:
            # Convert to one-hot if needed
            num_classes = y_pred.shape[1]
            y_true_onehot = np.zeros((y_true.shape[0], num_classes))
            y_true_onehot[np.arange(y_true.shape[0]), y_true.astype(int)] = 1
            y_true = y_true_onehot
        return -(y_true / y_pred) / y_true.shape[0]


class BinaryCrossEntropy(LossFunction):
    """Binary Cross-Entropy loss (for binary classification)"""
    
    def forward(self, y_true, y_pred):
        """Binary cross-entropy forward: -[y*log(p) + (1-y)*log(1-p)]"""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, y_true, y_pred):
        """Binary cross-entropy gradient: -(y/p - (1-y)/(1-p)) / n"""
        # Clip predictions to prevent division by 0
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]


class HuberLoss(LossFunction):
    """Huber loss (robust regression loss)"""
    
    def __init__(self, delta=1.0):
        self.delta = delta
    
    def forward(self, y_true, y_pred):
        """Huber loss forward"""
        error = y_true - y_pred
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        return np.mean(0.5 * quadratic**2 + self.delta * linear)
    
    def backward(self, y_true, y_pred):
        """Huber loss gradient"""
        error = y_true - y_pred
        return -np.where(np.abs(error) <= self.delta, error, self.delta * np.sign(error)) / y_true.shape[0]


class CategoricalCrossEntropy(LossFunction):
    """Categorical Cross-Entropy with Softmax (more numerically stable)"""
    
    def forward(self, y_true, y_pred_logits):
        """Forward pass using logits for numerical stability"""
        # Apply softmax to logits
        exp_logits = np.exp(y_pred_logits - np.max(y_pred_logits, axis=1, keepdims=True))
        y_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Compute cross-entropy
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        if y_true.ndim == 1:
            # Convert to one-hot if needed
            num_classes = y_pred.shape[1]
            y_true_onehot = np.zeros((y_true.shape[0], num_classes))
            y_true_onehot[np.arange(y_true.shape[0]), y_true.astype(int)] = 1
            y_true = y_true_onehot
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def backward(self, y_true, y_pred_logits):
        """Backward pass - gradient w.r.t. logits"""
        # Apply softmax to logits
        exp_logits = np.exp(y_pred_logits - np.max(y_pred_logits, axis=1, keepdims=True))
        y_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        if y_true.ndim == 1:
            # Convert to one-hot if needed
            num_classes = y_pred.shape[1]
            y_true_onehot = np.zeros((y_true.shape[0], num_classes))
            y_true_onehot[np.arange(y_true.shape[0]), y_true.astype(int)] = 1
            y_true = y_true_onehot
        
        # Gradient of softmax + cross-entropy: y_pred - y_true
        return (y_pred - y_true) / y_true.shape[0]


# Factory function for easy loss creation
def get_loss(name, **kwargs):
    """Get loss function by name"""
    losses = {
        'mse': MeanSquaredError,
        'mean_squared_error': MeanSquaredError,
        'mae': MeanAbsoluteError,
        'mean_absolute_error': MeanAbsoluteError,
        'crossentropy': CrossEntropy,
        'cross_entropy': CrossEntropy,
        'categorical_crossentropy': CategoricalCrossEntropy,
        'binary_crossentropy': BinaryCrossEntropy,
        'huber': HuberLoss
    }
    
    if name.lower() in losses:
        return losses[name.lower()](**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {name}") 