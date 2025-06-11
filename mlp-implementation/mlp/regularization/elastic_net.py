"""
Elastic Net Regularization implementation
"""

import numpy as np
from .base import Regularization


class ElasticNetRegularization(Regularization):
    """Elastic Net regularization - combination of L1 and L2"""
    
    def __init__(self, lambda_reg=0.01, l1_ratio=0.5):
        """
        Initialize Elastic Net regularization
        
        Args:
            lambda_reg: Total regularization strength
            l1_ratio: Ratio of L1 regularization (0.0 = pure L2, 1.0 = pure L1)
        """
        super().__init__(lambda_reg)
        self.l1_ratio = l1_ratio
        self.l2_ratio = 1 - l1_ratio
    
    def compute_penalty(self, weights):
        """
        Compute Elastic Net penalty: l1_ratio * L1 + l2_ratio * L2
        
        Args:
            weights: Weight matrix
            
        Returns:
            Elastic Net penalty value
        """
        l1_penalty = self.lambda_reg * self.l1_ratio * np.sum(np.abs(weights))
        l2_penalty = self.lambda_reg * self.l2_ratio * np.sum(np.square(weights)) / 2
        return l1_penalty + l2_penalty
    
    def compute_gradient(self, weights):
        """
        Compute Elastic Net gradient: l1_ratio * L1_grad + l2_ratio * L2_grad
        
        Args:
            weights: Weight matrix
            
        Returns:
            Elastic Net regularization gradient
        """
        l1_grad = self.lambda_reg * self.l1_ratio * np.sign(weights)
        l2_grad = self.lambda_reg * self.l2_ratio * weights
        return l1_grad + l2_grad
    
    def __repr__(self):
        return f"ElasticNetRegularization(lambda_reg={self.lambda_reg}, l1_ratio={self.l1_ratio})" 