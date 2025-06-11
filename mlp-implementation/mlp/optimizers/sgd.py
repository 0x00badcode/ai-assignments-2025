"""
Stochastic Gradient Descent (SGD) optimizer
"""

import numpy as np
from .base import Optimizer


class SGD(Optimizer):
    """Basic Stochastic Gradient Descent optimizer"""
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize SGD optimizer
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        super().__init__(learning_rate)
    
    def update(self, layer, gradients):
        """
        Update layer parameters using SGD
        
        Args:
            layer: Layer to update
            gradients: Dictionary of gradients for layer parameters
        """
        self.step_count += 1
        
        # Update weights
        if 'weights' in gradients:
            layer.weights -= self.learning_rate * gradients['weights']
        
        # Update bias
        if 'bias' in gradients and gradients['bias'] is not None:
            layer.bias -= self.learning_rate * gradients['bias']
        
        # Update batch norm parameters if present
        if hasattr(layer, 'gamma') and 'gamma' in gradients:
            layer.gamma -= self.learning_rate * gradients['gamma']
        
        if hasattr(layer, 'beta') and 'beta' in gradients:
            layer.beta -= self.learning_rate * gradients['beta']
    
    def __repr__(self):
        return f"SGD(learning_rate={self.learning_rate})" 