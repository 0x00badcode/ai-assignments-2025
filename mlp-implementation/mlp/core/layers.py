"""
Layer implementations for MLP
Includes Dense layer with various weight initialization methods
"""

import numpy as np
from abc import ABC, abstractmethod
from .activations import get_activation


class Layer(ABC):
    """Base class for neural network layers"""
    
    @abstractmethod
    def forward(self, x):
        """Forward pass"""
        pass
    
    @abstractmethod
    def backward(self, grad_output):
        """Backward pass"""
        pass
    
    @abstractmethod
    def get_params(self):
        """Get layer parameters"""
        pass
    
    @abstractmethod
    def set_params(self, params):
        """Set layer parameters"""
        pass


class Dense(Layer):
    """Fully connected (dense) layer"""
    
    def __init__(self, input_size, output_size, activation='linear', 
                 weight_init='xavier', bias_init='zeros', use_bias=True):
        """
        Initialize Dense layer
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            activation: Activation function name or instance
            weight_init: Weight initialization method
            bias_init: Bias initialization method
            use_bias: Whether to use bias
        """
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        
        # Initialize activation function
        if isinstance(activation, str):
            self.activation = get_activation(activation)
        else:
            self.activation = activation
        
        # Initialize weights and biases
        self.weights = self._init_weights(weight_init)
        if self.use_bias:
            self.bias = self._init_bias(bias_init)
        else:
            self.bias = None
        
        # Store for backward pass
        self.last_input = None
        self.last_z = None  # Pre-activation
        self.last_output = None  # Post-activation
    
    def _init_weights(self, method):
        """Initialize weights using specified method"""
        if method == 'zeros':
            return np.zeros((self.input_size, self.output_size))
        elif method == 'ones':
            return np.ones((self.input_size, self.output_size))
        elif method == 'random':
            return np.random.randn(self.input_size, self.output_size) * 0.01
        elif method == 'normal':
            return np.random.normal(0, 0.1, (self.input_size, self.output_size))
        elif method == 'uniform':
            return np.random.uniform(-0.1, 0.1, (self.input_size, self.output_size))
        elif method == 'xavier' or method == 'glorot':
            # Xavier/Glorot initialization
            limit = np.sqrt(6.0 / (self.input_size + self.output_size))
            return np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        elif method == 'xavier_normal':
            # Xavier normal initialization
            std = np.sqrt(2.0 / (self.input_size + self.output_size))
            return np.random.normal(0, std, (self.input_size, self.output_size))
        elif method == 'he' or method == 'kaiming':
            # He/Kaiming initialization (good for ReLU)
            std = np.sqrt(2.0 / self.input_size)
            return np.random.normal(0, std, (self.input_size, self.output_size))
        elif method == 'he_uniform':
            # He uniform initialization
            limit = np.sqrt(6.0 / self.input_size)
            return np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        elif method == 'lecun':
            # LeCun initialization
            std = np.sqrt(1.0 / self.input_size)
            return np.random.normal(0, std, (self.input_size, self.output_size))
        else:
            raise ValueError(f"Unknown weight initialization method: {method}")
    
    def _init_bias(self, method):
        """Initialize bias using specified method"""
        if method == 'zeros':
            return np.zeros(self.output_size)
        elif method == 'ones':
            return np.ones(self.output_size)
        elif method == 'random':
            return np.random.randn(self.output_size) * 0.01
        elif method == 'normal':
            return np.random.normal(0, 0.1, self.output_size)
        elif method == 'uniform':
            return np.random.uniform(-0.1, 0.1, self.output_size)
        else:
            raise ValueError(f"Unknown bias initialization method: {method}")
    
    def forward(self, x):
        """Forward pass through the layer"""
        self.last_input = x.copy()
        
        # Linear transformation: z = xW + b
        self.last_z = np.dot(x, self.weights)
        if self.use_bias:
            self.last_z += self.bias
        
        # Apply activation function
        self.last_output = self.activation.forward(self.last_z)
        
        return self.last_output
    
    def backward(self, grad_output):
        """Backward pass through the layer"""
        # Gradient w.r.t. activation input
        grad_activation = self.activation.backward(self.last_z)
        grad_z = grad_output * grad_activation
        
        # Gradient w.r.t. weights
        grad_weights = np.dot(self.last_input.T, grad_z)
        
        # Gradient w.r.t. bias
        if self.use_bias:
            grad_bias = np.sum(grad_z, axis=0)
        else:
            grad_bias = None
        
        # Gradient w.r.t. input (for previous layer)
        grad_input = np.dot(grad_z, self.weights.T)
        
        # Store gradients
        self.grad_weights = grad_weights
        self.grad_bias = grad_bias
        
        return grad_input
    
    def get_params(self):
        """Get layer parameters"""
        params = {'weights': self.weights.copy()}
        if self.use_bias:
            params['bias'] = self.bias.copy()
        return params
    
    def set_params(self, params):
        """Set layer parameters"""
        self.weights = params['weights'].copy()
        if self.use_bias and 'bias' in params:
            self.bias = params['bias'].copy()
    
    def get_gradients(self):
        """Get parameter gradients"""
        grads = {'weights': self.grad_weights}
        if self.use_bias:
            grads['bias'] = self.grad_bias
        return grads
    
    def update_params(self, learning_rate):
        """Simple parameter update (basic SGD)"""
        self.weights -= learning_rate * self.grad_weights
        if self.use_bias:
            self.bias -= learning_rate * self.grad_bias
    
    def __repr__(self):
        return f"Dense({self.input_size}, {self.output_size}, activation={self.activation.__class__.__name__})"


class Dropout(Layer):
    """Dropout layer for regularization"""
    
    def __init__(self, rate=0.5):
        """
        Initialize Dropout layer
        
        Args:
            rate: Dropout rate (fraction of inputs to drop)
        """
        self.rate = rate
        self.training = True
        self.mask = None
    
    def forward(self, x):
        """Forward pass through dropout layer"""
        if self.training and self.rate > 0:
            # Create dropout mask
            self.mask = np.random.binomial(1, 1-self.rate, x.shape) / (1-self.rate)
            return x * self.mask
        else:
            return x
    
    def backward(self, grad_output):
        """Backward pass through dropout layer"""
        if self.training and self.rate > 0:
            return grad_output * self.mask
        else:
            return grad_output
    
    def get_params(self):
        """Dropout has no parameters"""
        return {}
    
    def set_params(self, params):
        """Dropout has no parameters"""
        pass
    
    def set_training(self, training):
        """Set training mode"""
        self.training = training
    
    def __repr__(self):
        return f"Dropout(rate={self.rate})"


class BatchNormalization(Layer):
    """Batch Normalization layer"""
    
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        """
        Initialize Batch Normalization layer
        
        Args:
            num_features: Number of features
            momentum: Momentum for running statistics
            epsilon: Small value for numerical stability
        """
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones(num_features)  # Scale
        self.beta = np.zeros(num_features)  # Shift
        
        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Training mode
        self.training = True
        
        # Store for backward pass
        self.last_input = None
        self.last_normalized = None
        self.last_var = None
        self.last_mean = None
    
    def forward(self, x):
        """Forward pass through batch normalization"""
        self.last_input = x.copy()
        
        if self.training:
            # Compute batch statistics
            self.last_mean = np.mean(x, axis=0)
            self.last_var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.last_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.last_var
            
            # Normalize
            self.last_normalized = (x - self.last_mean) / np.sqrt(self.last_var + self.epsilon)
        else:
            # Use running statistics for inference
            self.last_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Scale and shift
        return self.gamma * self.last_normalized + self.beta
    
    def backward(self, grad_output):
        """Backward pass through batch normalization"""
        batch_size = self.last_input.shape[0]
        
        # Gradients w.r.t. gamma and beta
        self.grad_gamma = np.sum(grad_output * self.last_normalized, axis=0)
        self.grad_beta = np.sum(grad_output, axis=0)
        
        # Gradient w.r.t. normalized input
        grad_normalized = grad_output * self.gamma
        
        # Gradient w.r.t. input
        if self.training:
            grad_var = np.sum(grad_normalized * (self.last_input - self.last_mean), axis=0) * \
                      -0.5 * np.power(self.last_var + self.epsilon, -1.5)
            
            grad_mean = np.sum(grad_normalized * -1.0 / np.sqrt(self.last_var + self.epsilon), axis=0) + \
                       grad_var * np.sum(-2.0 * (self.last_input - self.last_mean), axis=0) / batch_size
            
            grad_input = grad_normalized / np.sqrt(self.last_var + self.epsilon) + \
                        grad_var * 2.0 * (self.last_input - self.last_mean) / batch_size + \
                        grad_mean / batch_size
        else:
            grad_input = grad_normalized / np.sqrt(self.running_var + self.epsilon)
        
        return grad_input
    
    def get_params(self):
        """Get layer parameters"""
        return {
            'gamma': self.gamma.copy(),
            'beta': self.beta.copy(),
            'running_mean': self.running_mean.copy(),
            'running_var': self.running_var.copy()
        }
    
    def set_params(self, params):
        """Set layer parameters"""
        self.gamma = params['gamma'].copy()
        self.beta = params['beta'].copy()
        if 'running_mean' in params:
            self.running_mean = params['running_mean'].copy()
        if 'running_var' in params:
            self.running_var = params['running_var'].copy()
    
    def get_gradients(self):
        """Get parameter gradients"""
        return {
            'gamma': self.grad_gamma,
            'beta': self.grad_beta
        }
    
    def set_training(self, training):
        """Set training mode"""
        self.training = training
    
    def __repr__(self):
        return f"BatchNormalization({self.num_features})" 