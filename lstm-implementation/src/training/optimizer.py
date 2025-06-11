"""
Optimizers for training LSTM models.
Includes Adam, SGD, and other optimization algorithms.
"""
import numpy as np
from ..core.utils import clip_gradients

class Optimizer:
    """Base optimizer class."""
    
    def __init__(self, learning_rate=0.001):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Learning rate
        """
        self.learning_rate = learning_rate
        self.step_count = 0
    
    def update(self, parameters, gradients):
        """
        Update parameters using gradients.
        
        Args:
            parameters: List of parameter arrays
            gradients: List of gradient arrays
        """
        raise NotImplementedError("Subclasses must implement update method")
    
    def zero_grad(self, gradients):
        """Zero out gradients."""
        for grad in gradients:
            if grad is not None:
                grad.fill(0)

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate=0.01, momentum=0.0, weight_decay=0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum factor
            weight_decay: Weight decay (L2 regularization)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = None
    
    def update(self, parameters, gradients):
        """Update parameters using SGD with momentum."""
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in parameters]
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            if grad is not None:
                # Add weight decay
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param
                
                # Update velocity
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
                
                # Update parameter
                param += self.velocity[i]
        
        self.step_count += 1

class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            eps: Small constant for numerical stability
            weight_decay: Weight decay (L2 regularization)
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        
        # State variables
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
    
    def update(self, parameters, gradients):
        """Update parameters using Adam."""
        if self.m is None:
            self.m = [np.zeros_like(param) for param in parameters]
            self.v = [np.zeros_like(param) for param in parameters]
        
        self.step_count += 1
        
        # Bias correction factors
        bias_correction1 = 1 - self.beta1 ** self.step_count
        bias_correction2 = 1 - self.beta2 ** self.step_count
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            if grad is not None:
                # Add weight decay
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param
                
                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                
                # Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / bias_correction1
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / bias_correction2
                
                # Update parameter
                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self, learning_rate=0.001, alpha=0.99, eps=1e-8, weight_decay=0.0):
        """
        Initialize RMSprop optimizer.
        
        Args:
            learning_rate: Learning rate
            alpha: Smoothing constant
            eps: Small constant for numerical stability
            weight_decay: Weight decay (L2 regularization)
        """
        super().__init__(learning_rate)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.v = None  # Moving average of squared gradients
    
    def update(self, parameters, gradients):
        """Update parameters using RMSprop."""
        if self.v is None:
            self.v = [np.zeros_like(param) for param in parameters]
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            if grad is not None:
                # Add weight decay
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param
                
                # Update moving average of squared gradients
                self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (grad ** 2)
                
                # Update parameter
                param -= self.learning_rate * grad / (np.sqrt(self.v[i]) + self.eps)
        
        self.step_count += 1

class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay)."""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        """
        Initialize AdamW optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            eps: Small constant for numerical stability
            weight_decay: Weight decay factor
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        
        # State variables
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
    
    def update(self, parameters, gradients):
        """Update parameters using AdamW."""
        if self.m is None:
            self.m = [np.zeros_like(param) for param in parameters]
            self.v = [np.zeros_like(param) for param in parameters]
        
        self.step_count += 1
        
        # Bias correction factors
        bias_correction1 = 1 - self.beta1 ** self.step_count
        bias_correction2 = 1 - self.beta2 ** self.step_count
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            if grad is not None:
                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                
                # Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / bias_correction1
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / bias_correction2
                
                # Update parameter with decoupled weight decay
                param -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.eps) + 
                                             self.weight_decay * param)

class LearningRateScheduler:
    """Learning rate scheduler."""
    
    def __init__(self, optimizer, schedule_type='constant', **kwargs):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            schedule_type: Type of schedule ('constant', 'step', 'cosine', 'exponential')
            **kwargs: Additional arguments for specific schedules
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.initial_lr = optimizer.learning_rate
        
        # Schedule-specific parameters
        if schedule_type == 'step':
            self.step_size = kwargs.get('step_size', 1000)
            self.gamma = kwargs.get('gamma', 0.1)
        elif schedule_type == 'exponential':
            self.gamma = kwargs.get('gamma', 0.95)
        elif schedule_type == 'cosine':
            self.T_max = kwargs.get('T_max', 1000)
            self.eta_min = kwargs.get('eta_min', 0.0)
    
    def step(self, step_num=None):
        """Update learning rate based on step number."""
        if step_num is None:
            step_num = self.optimizer.step_count
        
        if self.schedule_type == 'constant':
            lr = self.initial_lr
        elif self.schedule_type == 'step':
            lr = self.initial_lr * (self.gamma ** (step_num // self.step_size))
        elif self.schedule_type == 'exponential':
            lr = self.initial_lr * (self.gamma ** step_num)
        elif self.schedule_type == 'cosine':
            lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                 (1 + np.cos(np.pi * step_num / self.T_max)) / 2
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        self.optimizer.learning_rate = lr
        return lr

class GradientClipper:
    """Gradient clipping utility."""
    
    def __init__(self, max_norm=5.0, norm_type=2):
        """
        Initialize gradient clipper.
        
        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of norm (1, 2, or 'inf')
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip(self, gradients):
        """Clip gradients by global norm."""
        if self.norm_type == 2:
            return clip_gradients(gradients, self.max_norm)
        elif self.norm_type == 1:
            # L1 norm clipping
            total_norm = sum(np.sum(np.abs(grad)) for grad in gradients if grad is not None)
            if total_norm > self.max_norm:
                clip_coef = self.max_norm / (total_norm + 1e-6)
                for i, grad in enumerate(gradients):
                    if grad is not None:
                        gradients[i] = grad * clip_coef
        elif self.norm_type == 'inf':
            # Infinity norm clipping
            for i, grad in enumerate(gradients):
                if grad is not None:
                    gradients[i] = np.clip(grad, -self.max_norm, self.max_norm)
        
        return gradients

def create_optimizer(optimizer_type, **kwargs):
    """
    Factory function to create optimizer.
    
    Args:
        optimizer_type: Type of optimizer ('sgd', 'adam', 'rmsprop', 'adamw')
        **kwargs: Optimizer-specific arguments
    
    Returns:
        Optimizer instance
    """
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'sgd':
        return SGD(**kwargs)
    elif optimizer_type == 'adam':
        return Adam(**kwargs)
    elif optimizer_type == 'rmsprop':
        return RMSprop(**kwargs)
    elif optimizer_type == 'adamw':
        return AdamW(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def create_scheduler(optimizer, scheduler_type, **kwargs):
    """
    Factory function to create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        **kwargs: Scheduler-specific arguments
    
    Returns:
        LearningRateScheduler instance
    """
    return LearningRateScheduler(optimizer, scheduler_type, **kwargs) 