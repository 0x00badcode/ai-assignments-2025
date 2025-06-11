"""
SGD with Momentum optimizer
"""

import numpy as np
from .base import Optimizer


class Momentum(Optimizer):
    """SGD with Momentum optimizer"""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Initialize Momentum optimizer
        
        Args:
            learning_rate: Learning rate for parameter updates
            momentum: Momentum factor
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, layer, gradients):
        """
        Update layer parameters using SGD with momentum
        
        Args:
            layer: Layer to update
            gradients: Dictionary of gradients for layer parameters
        """
        self.step_count += 1
        layer_id = id(layer)
        
        # Initialize velocities for this layer if not exists
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {}
        
        # Update weights
        if 'weights' in gradients:
            if 'weights' not in self.velocities[layer_id]:
                self.velocities[layer_id]['weights'] = np.zeros_like(layer.weights)
            
            # v = momentum * v + learning_rate * gradient
            self.velocities[layer_id]['weights'] = (
                self.momentum * self.velocities[layer_id]['weights'] + 
                self.learning_rate * gradients['weights']
            )
            
            # Update weights: w = w - v
            layer.weights -= self.velocities[layer_id]['weights']
        
        # Update bias
        if 'bias' in gradients and gradients['bias'] is not None:
            if 'bias' not in self.velocities[layer_id]:
                self.velocities[layer_id]['bias'] = np.zeros_like(layer.bias)
            
            self.velocities[layer_id]['bias'] = (
                self.momentum * self.velocities[layer_id]['bias'] + 
                self.learning_rate * gradients['bias']
            )
            
            layer.bias -= self.velocities[layer_id]['bias']
        
        # Update batch norm parameters if present
        if hasattr(layer, 'gamma') and 'gamma' in gradients:
            if 'gamma' not in self.velocities[layer_id]:
                self.velocities[layer_id]['gamma'] = np.zeros_like(layer.gamma)
            
            self.velocities[layer_id]['gamma'] = (
                self.momentum * self.velocities[layer_id]['gamma'] + 
                self.learning_rate * gradients['gamma']
            )
            
            layer.gamma -= self.velocities[layer_id]['gamma']
        
        if hasattr(layer, 'beta') and 'beta' in gradients:
            if 'beta' not in self.velocities[layer_id]:
                self.velocities[layer_id]['beta'] = np.zeros_like(layer.beta)
            
            self.velocities[layer_id]['beta'] = (
                self.momentum * self.velocities[layer_id]['beta'] + 
                self.learning_rate * gradients['beta']
            )
            
            layer.beta -= self.velocities[layer_id]['beta']
    
    def get_config(self):
        """Get optimizer configuration"""
        config = super().get_config()
        config['momentum'] = self.momentum
        return config
    
    def set_config(self, config):
        """Set optimizer configuration"""
        super().set_config(config)
        self.momentum = config.get('momentum', self.momentum)
    
    def __repr__(self):
        return f"Momentum(learning_rate={self.learning_rate}, momentum={self.momentum})" 