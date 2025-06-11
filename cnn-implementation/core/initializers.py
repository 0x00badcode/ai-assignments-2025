import numpy as np
from typing import Tuple, Union

def glorot_uniform(shape: Tuple[int, ...]) -> np.ndarray:
    """Initialize weights using Glorot/Xavier uniform initialization.
    
    The weights are drawn from a uniform distribution within [-limit, limit]
    where limit is sqrt(6 / (fan_in + fan_out)).
    
    Args:
        shape: Shape of the weight matrix to initialize
        
    Returns:
        Initialized weights
    """
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    limit = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)

def glorot_normal(shape: Tuple[int, ...]) -> np.ndarray:
    """Initialize weights using Glorot/Xavier normal initialization.
    
    The weights are drawn from a normal distribution with mean 0
    and standard deviation sqrt(2 / (fan_in + fan_out)).
    
    Args:
        shape: Shape of the weight matrix to initialize
        
    Returns:
        Initialized weights
    """
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    std = np.sqrt(2. / (fan_in + fan_out))
    return np.random.normal(0., std, shape)

def he_normal(shape: Tuple[int, ...]) -> np.ndarray:
    """Initialize weights using He normal initialization.
    
    The weights are drawn from a normal distribution with mean 0
    and standard deviation sqrt(2 / fan_in).
    
    Args:
        shape: Shape of the weight matrix to initialize
        
    Returns:
        Initialized weights
    """
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    std = np.sqrt(2. / fan_in)
    return np.random.normal(0., std, shape)

def he_uniform(shape: Tuple[int, ...]) -> np.ndarray:
    """Initialize weights using He uniform initialization.
    
    The weights are drawn from a uniform distribution within [-limit, limit]
    where limit is sqrt(6 / fan_in).
    
    Args:
        shape: Shape of the weight matrix to initialize
        
    Returns:
        Initialized weights
    """
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    limit = np.sqrt(6. / fan_in)
    return np.random.uniform(-limit, limit, shape)

def get_initializer(name: str):
    """Get initializer function by name.
    
    Args:
        name: Name of the initializer ('glorot_uniform', 'glorot_normal',
              'he_normal', 'he_uniform')
              
    Returns:
        Initializer function
    """
    initializers = {
        'glorot_uniform': glorot_uniform,
        'glorot_normal': glorot_normal,
        'he_normal': he_normal,
        'he_uniform': he_uniform
    }
    return initializers.get(name, glorot_uniform)  # Default to glorot_uniform