"""
L2 Regularization (Ridge) implementation
"""

import numpy as np
from .base import Regularization


class L2Regularization(Regularization):
    """L2 regularization (Ridge) - promotes smaller weights"""
    
    def compute_penalty(self, weights):
        """
        Compute L2 penalty: lambda * sum(weights^2) / 2
        
        Args:
            weights: Weight matrix
            
        Returns:
            L2 penalty value
        """
        return self.lambda_reg * np.sum(np.square(weights)) / 2
    
    def compute_gradient(self, weights):
        """
        Compute L2 gradient: lambda * weights
        
        Args:
            weights: Weight matrix
            
        Returns:
            L2 regularization gradient
        """
        return self.lambda_reg * weights 