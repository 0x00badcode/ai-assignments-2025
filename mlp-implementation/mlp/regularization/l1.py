"""
L1 Regularization (Lasso) implementation
"""

import numpy as np
from .base import Regularization


class L1Regularization(Regularization):
    """L1 regularization (Lasso) - promotes sparsity"""
    
    def compute_penalty(self, weights):
        """
        Compute L1 penalty: lambda * sum(|weights|)
        
        Args:
            weights: Weight matrix
            
        Returns:
            L1 penalty value
        """
        return self.lambda_reg * np.sum(np.abs(weights))
    
    def compute_gradient(self, weights):
        """
        Compute L1 gradient: lambda * sign(weights)
        
        Args:
            weights: Weight matrix
            
        Returns:
            L1 regularization gradient
        """
        return self.lambda_reg * np.sign(weights) 