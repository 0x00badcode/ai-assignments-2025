"""
Base Regularization class for MLP Implementation
"""

import numpy as np
from abc import ABC, abstractmethod


class Regularization(ABC):
    """Base class for regularization techniques"""
    
    def __init__(self, lambda_reg=0.01):
        """
        Initialize regularization
        
        Args:
            lambda_reg: Regularization strength
        """
        self.lambda_reg = lambda_reg
    
    @abstractmethod
    def compute_penalty(self, weights):
        """
        Compute regularization penalty
        
        Args:
            weights: Weight matrix
            
        Returns:
            Regularization penalty value
        """
        pass
    
    @abstractmethod
    def compute_gradient(self, weights):
        """
        Compute regularization gradient
        
        Args:
            weights: Weight matrix
            
        Returns:
            Regularization gradient
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(lambda_reg={self.lambda_reg})" 