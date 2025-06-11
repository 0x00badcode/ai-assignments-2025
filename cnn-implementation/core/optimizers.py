import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Type

class Optimizer(ABC):
    """Abstract base class for optimizers."""
    
    def __init__(self, learning_rate: float = 0.01):
        """Initialize optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        self.learning_rate = learning_rate
        self.iterations = 0
    
    @abstractmethod
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update parameters using their gradients.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of parameter gradients
            
        Returns:
            Updated parameters
        """
        pass

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
            momentum: Momentum coefficient
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            
            # Update velocity
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            
            # Update parameters
            params[key] += self.velocity[key]
        
        self.iterations += 1
        return params

class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, decay_rate: float = 0.99, epsilon: float = 1e-8):
        """Initialize RMSprop optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
            decay_rate: Decay rate for moving average
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for key in params:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            
            # Update cache
            self.cache[key] = self.decay_rate * self.cache[key] + \
                            (1 - self.decay_rate) * np.square(grads[key])
            
            # Update parameters
            params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        
        self.iterations += 1
        return params

class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        """Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(grads[key])
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** (self.iterations + 1))
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** (self.iterations + 1))
            
            # Update parameters
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        self.iterations += 1
        return params

# Dictionary mapping optimizer names to their classes
OPTIMIZERS: Dict[str, Type[Optimizer]] = {
    'sgd': SGD,
    'rmsprop': RMSprop,
    'adam': Adam
}

def get_optimizer(name: str, **kwargs) -> Optimizer:
    """Get optimizer by name.
    
    Args:
        name: Name of the optimizer
        **kwargs: Additional optimizer parameters
        
    Returns:
        Optimizer instance
        
    Raises:
        ValueError: If optimizer is not found
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    return OPTIMIZERS[name](**kwargs)