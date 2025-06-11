"""
Adam optimizer
"""

import numpy as np
from .base import Optimizer


class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer
        
        Args:
            learning_rate: Learning rate for parameter updates
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small value for numerical stability
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
    
    def update(self, layer, gradients):
        """
        Update layer parameters using Adam
        
        Args:
            layer: Layer to update
            gradients: Dictionary of gradients for layer parameters
        """
        self.step_count += 1
        layer_id = id(layer)
        
        # Initialize moments for this layer if not exists
        if layer_id not in self.m:
            self.m[layer_id] = {}
            self.v[layer_id] = {}
        
        # Bias correction factors
        bias_correction1 = 1 - self.beta1 ** self.step_count
        bias_correction2 = 1 - self.beta2 ** self.step_count
        
        # Update weights
        if 'weights' in gradients:
            if 'weights' not in self.m[layer_id]:
                self.m[layer_id]['weights'] = np.zeros_like(layer.weights)
                self.v[layer_id]['weights'] = np.zeros_like(layer.weights)
            
            # Update first moment: m = beta1 * m + (1 - beta1) * gradient
            self.m[layer_id]['weights'] = (
                self.beta1 * self.m[layer_id]['weights'] + 
                (1 - self.beta1) * gradients['weights']
            )
            
            # Update second moment: v = beta2 * v + (1 - beta2) * gradient^2
            self.v[layer_id]['weights'] = (
                self.beta2 * self.v[layer_id]['weights'] + 
                (1 - self.beta2) * np.square(gradients['weights'])
            )
            
            # Bias-corrected moments
            m_hat = self.m[layer_id]['weights'] / bias_correction1
            v_hat = self.v[layer_id]['weights'] / bias_correction2
            
            # Update weights: w = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
            layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Update bias
        if 'bias' in gradients and gradients['bias'] is not None:
            if 'bias' not in self.m[layer_id]:
                self.m[layer_id]['bias'] = np.zeros_like(layer.bias)
                self.v[layer_id]['bias'] = np.zeros_like(layer.bias)
            
            self.m[layer_id]['bias'] = (
                self.beta1 * self.m[layer_id]['bias'] + 
                (1 - self.beta1) * gradients['bias']
            )
            
            self.v[layer_id]['bias'] = (
                self.beta2 * self.v[layer_id]['bias'] + 
                (1 - self.beta2) * np.square(gradients['bias'])
            )
            
            m_hat = self.m[layer_id]['bias'] / bias_correction1
            v_hat = self.v[layer_id]['bias'] / bias_correction2
            
            layer.bias -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Update batch norm parameters if present
        if hasattr(layer, 'gamma') and 'gamma' in gradients:
            if 'gamma' not in self.m[layer_id]:
                self.m[layer_id]['gamma'] = np.zeros_like(layer.gamma)
                self.v[layer_id]['gamma'] = np.zeros_like(layer.gamma)
            
            self.m[layer_id]['gamma'] = (
                self.beta1 * self.m[layer_id]['gamma'] + 
                (1 - self.beta1) * gradients['gamma']
            )
            
            self.v[layer_id]['gamma'] = (
                self.beta2 * self.v[layer_id]['gamma'] + 
                (1 - self.beta2) * np.square(gradients['gamma'])
            )
            
            m_hat = self.m[layer_id]['gamma'] / bias_correction1
            v_hat = self.v[layer_id]['gamma'] / bias_correction2
            
            layer.gamma -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        if hasattr(layer, 'beta') and 'beta' in gradients:
            if 'beta' not in self.m[layer_id]:
                self.m[layer_id]['beta'] = np.zeros_like(layer.beta)
                self.v[layer_id]['beta'] = np.zeros_like(layer.beta)
            
            self.m[layer_id]['beta'] = (
                self.beta1 * self.m[layer_id]['beta'] + 
                (1 - self.beta1) * gradients['beta']
            )
            
            self.v[layer_id]['beta'] = (
                self.beta2 * self.v[layer_id]['beta'] + 
                (1 - self.beta2) * np.square(gradients['beta'])
            )
            
            m_hat = self.m[layer_id]['beta'] / bias_correction1
            v_hat = self.v[layer_id]['beta'] / bias_correction2
            
            layer.beta -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def get_config(self):
        """Get optimizer configuration"""
        config = super().get_config()
        config.update({
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon
        })
        return config
    
    def set_config(self, config):
        """Set optimizer configuration"""
        super().set_config(config)
        self.beta1 = config.get('beta1', self.beta1)
        self.beta2 = config.get('beta2', self.beta2)
        self.epsilon = config.get('epsilon', self.epsilon)
    
    def __repr__(self):
        return f"Adam(learning_rate={self.learning_rate}, beta1={self.beta1}, beta2={self.beta2}, epsilon={self.epsilon})" 