"""
RMSProp optimizer
"""

import numpy as np
from .base import Optimizer


class RMSProp(Optimizer):
    """RMSProp optimizer"""
    
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        """
        Initialize RMSProp optimizer
        
        Args:
            learning_rate: Learning rate for parameter updates
            rho: Decay rate for moving average of squared gradients
            epsilon: Small value for numerical stability
        """
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.cache = {}
    
    def update(self, layer, gradients):
        """
        Update layer parameters using RMSProp
        
        Args:
            layer: Layer to update
            gradients: Dictionary of gradients for layer parameters
        """
        self.step_count += 1
        layer_id = id(layer)
        
        # Initialize cache for this layer if not exists
        if layer_id not in self.cache:
            self.cache[layer_id] = {}
        
        # Update weights
        if 'weights' in gradients:
            if 'weights' not in self.cache[layer_id]:
                self.cache[layer_id]['weights'] = np.zeros_like(layer.weights)
            
            # cache = rho * cache + (1 - rho) * gradient^2
            self.cache[layer_id]['weights'] = (
                self.rho * self.cache[layer_id]['weights'] + 
                (1 - self.rho) * np.square(gradients['weights'])
            )
            
            # Update weights: w = w - learning_rate * gradient / sqrt(cache + epsilon)
            layer.weights -= (
                self.learning_rate * gradients['weights'] / 
                (np.sqrt(self.cache[layer_id]['weights']) + self.epsilon)
            )
        
        # Update bias
        if 'bias' in gradients and gradients['bias'] is not None:
            if 'bias' not in self.cache[layer_id]:
                self.cache[layer_id]['bias'] = np.zeros_like(layer.bias)
            
            self.cache[layer_id]['bias'] = (
                self.rho * self.cache[layer_id]['bias'] + 
                (1 - self.rho) * np.square(gradients['bias'])
            )
            
            layer.bias -= (
                self.learning_rate * gradients['bias'] / 
                (np.sqrt(self.cache[layer_id]['bias']) + self.epsilon)
            )
        
        # Update batch norm parameters if present
        if hasattr(layer, 'gamma') and 'gamma' in gradients:
            if 'gamma' not in self.cache[layer_id]:
                self.cache[layer_id]['gamma'] = np.zeros_like(layer.gamma)
            
            self.cache[layer_id]['gamma'] = (
                self.rho * self.cache[layer_id]['gamma'] + 
                (1 - self.rho) * np.square(gradients['gamma'])
            )
            
            layer.gamma -= (
                self.learning_rate * gradients['gamma'] / 
                (np.sqrt(self.cache[layer_id]['gamma']) + self.epsilon)
            )
        
        if hasattr(layer, 'beta') and 'beta' in gradients:
            if 'beta' not in self.cache[layer_id]:
                self.cache[layer_id]['beta'] = np.zeros_like(layer.beta)
            
            self.cache[layer_id]['beta'] = (
                self.rho * self.cache[layer_id]['beta'] + 
                (1 - self.rho) * np.square(gradients['beta'])
            )
            
            layer.beta -= (
                self.learning_rate * gradients['beta'] / 
                (np.sqrt(self.cache[layer_id]['beta']) + self.epsilon)
            )
    
    def get_config(self):
        """Get optimizer configuration"""
        config = super().get_config()
        config.update({
            'rho': self.rho,
            'epsilon': self.epsilon
        })
        return config
    
    def set_config(self, config):
        """Set optimizer configuration"""
        super().set_config(config)
        self.rho = config.get('rho', self.rho)
        self.epsilon = config.get('epsilon', self.epsilon)
    
    def __repr__(self):
        return f"RMSProp(learning_rate={self.learning_rate}, rho={self.rho}, epsilon={self.epsilon})" 