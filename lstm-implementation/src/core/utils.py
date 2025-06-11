"""
Utility functions for LSTM implementation.
"""
import numpy as np

def xavier_uniform(shape, gain=1.0):
    """Xavier uniform initialization."""
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)

def he_uniform(shape):
    """He uniform initialization for ReLU activations."""
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, shape)

def orthogonal_init(shape, gain=1.0):
    """Orthogonal initialization."""
    if len(shape) < 2:
        raise ValueError("Orthogonal initialization requires at least 2D tensor")
    
    rows, cols = shape[0], np.prod(shape[1:])
    
    # Generate random matrix
    a = np.random.randn(rows, cols)
    
    # SVD decomposition
    u, _, vh = np.linalg.svd(a, full_matrices=False)
    
    # Pick the one with the correct shape
    q = u if u.shape == (rows, cols) else vh
    q = q.reshape(shape)
    
    return gain * q

def clip_gradients(grads, max_norm=5.0):
    """Clip gradients by global norm."""
    total_norm = 0
    for grad in grads:
        if grad is not None:
            total_norm += np.sum(grad ** 2)
    
    total_norm = np.sqrt(total_norm)
    
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for i, grad in enumerate(grads):
            if grad is not None:
                grads[i] = grad * clip_coef
    
    return grads

def dropout_mask(shape, dropout_rate, training=True):
    """Generate dropout mask."""
    if not training or dropout_rate == 0:
        return np.ones(shape)
    
    mask = np.random.binomial(1, 1 - dropout_rate, shape).astype(np.float32)
    return mask / (1 - dropout_rate)  # Scale to maintain expected value

def recurrent_dropout_mask(shape, dropout_rate, training=True):
    """Generate recurrent dropout mask (same across time steps)."""
    if not training or dropout_rate == 0:
        return np.ones(shape)
    
    # Same mask for all time steps
    mask = np.random.binomial(1, 1 - dropout_rate, shape).astype(np.float32)
    return mask / (1 - dropout_rate)

def one_hot_encode(indices, num_classes):
    """Convert indices to one-hot encoded vectors."""
    batch_size = indices.shape[0]
    one_hot = np.zeros((batch_size, num_classes))
    one_hot[np.arange(batch_size), indices] = 1
    return one_hot

def pad_sequences(sequences, max_length=None, padding='post', truncating='post', value=0):
    """Pad sequences to the same length."""
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        seq = np.array(seq)
        
        # Truncate if necessary
        if len(seq) > max_length:
            if truncating == 'post':
                seq = seq[:max_length]
            else:  # 'pre'
                seq = seq[-max_length:]
        
        # Pad if necessary
        if len(seq) < max_length:
            pad_width = max_length - len(seq)
            if padding == 'post':
                seq = np.concatenate([seq, np.full(pad_width, value)])
            else:  # 'pre'
                seq = np.concatenate([np.full(pad_width, value), seq])
        
        padded.append(seq)
    
    return np.array(padded)

def create_mask(sequences, pad_value=0):
    """Create attention mask for padded sequences."""
    return (sequences != pad_value).astype(np.float32)

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b) 