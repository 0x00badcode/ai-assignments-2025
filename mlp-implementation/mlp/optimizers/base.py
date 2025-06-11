"""
Base Optimizer class for MLP Implementation
"""

import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Base class for optimizers"""
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize optimizer
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        self.learning_rate = learning_rate
        self.step_count = 0
    
    @abstractmethod
    def update(self, layer, gradients):
        """
        Update layer parameters using gradients
        
        Args:
            layer: Layer to update
            gradients: Dictionary of gradients for layer parameters
        """
        pass
    
    def zero_gradients(self, layer):
        """Zero out gradients (optional, some optimizers may need this)"""
        pass
    
    def get_config(self):
        """Get optimizer configuration"""
        return {
            'learning_rate': self.learning_rate,
            'step_count': self.step_count
        }
    
    def set_config(self, config):
        """Set optimizer configuration"""
        self.learning_rate = config.get('learning_rate', self.learning_rate)
        self.step_count = config.get('step_count', self.step_count) 