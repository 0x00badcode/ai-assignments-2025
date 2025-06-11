from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Union, Optional
from ..matrix import Matrix
from ..activations import Activation, get_activation
from ..initializers import get_initializer

class Layer(ABC):
    """Abstract base class for all neural network layers."""
    
    def __init__(self, l1_lambda: float = 0.0, l2_lambda: float = 0.0):
        self.input = None
        self.output = None
        self.gradients = {}
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
    
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass computation.
        
        Args:
            inputs: Input data
            
        Returns:
            Layer outputs
        """
        pass
    
    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass computation.
        
        Args:
            grad: Gradient from the next layer
            
        Returns:
            Gradient with respect to the input
        """
        pass
    
    @property
    def parameters(self) -> dict:
        """Get layer parameters."""
        return {}
    
    @property
    def parameter_gradients(self) -> dict:
        """Get gradients with respect to parameters."""
        return self.gradients

class Conv2D(Layer):
    """2D Convolutional Layer."""
    
    def __init__(self, filters: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1, padding: str = 'valid',
                 activation: Optional[str] = None,
                 kernel_initializer: str = 'glorot_uniform',
                 l1_lambda: float = 0.0, l2_lambda: float = 0.0):
        """Initialize Conv2D layer.
        
        Args:
            filters: Number of filters
            kernel_size: Size of the convolution kernel
            stride: Convolution stride
            padding: Padding mode ('valid' or 'same')
            activation: Activation function name
        """
        super().__init__(l1_lambda, l2_lambda)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.activation = get_activation(activation) if activation else None
        self.kernel_initializer = kernel_initializer
        
        # Initialize parameters
        self.weights = None
        self.bias = None
        self.initialized = False
    
    def initialize(self, input_shape: Tuple[int, ...]):
        """Initialize layer parameters."""
        if not self.initialized:
            _, channels, _, _ = input_shape
            initializer = get_initializer(self.kernel_initializer)
            self.weights = initializer((self.filters, channels, *self.kernel_size))
            self.bias = np.zeros(self.filters)
            self.initialized = True
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input = inputs
        self.initialize(inputs.shape)
        
        N, C, H, W = inputs.shape
        F, _, KH, KW = self.weights.shape
        
        # Calculate padding
        if self.padding == 'same':
            pad_h = (KH - 1) // 2
            pad_w = (KW - 1) // 2
            inputs = np.pad(inputs, ((0,0), (0,0), (pad_h,pad_h), (pad_w,pad_w)))
        
        # Calculate output dimensions
        out_h = (H - KH + 2 * (self.padding == 'same') * ((KH - 1) // 2)) // self.stride + 1
        out_w = (W - KW + 2 * (self.padding == 'same') * ((KW - 1) // 2)) // self.stride + 1
        
        # Convert input to column matrix
        x_col = Matrix(inputs).im2col((KH, KW), self.stride)
        w_col = self.weights.reshape(F, -1)
        
        # Compute convolution
        out = np.dot(w_col, x_col.data.T).T
        out = out.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)
        out = out + self.bias.reshape(1, F, 1, 1)
        
        if self.activation:
            out = self.activation.forward(out)
        
        self.output = out
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.activation:
            grad = self.activation.backward(self.output) * grad
        
        N, F, H, W = grad.shape
        _, C, KH, KW = self.weights.shape
        
        # Compute bias gradient
        self.gradients['bias'] = np.sum(grad, axis=(0,2,3))
        
        # Compute weights gradient
        x_col = Matrix(self.input).im2col((KH, KW), self.stride)
        grad_col = grad.transpose(0,2,3,1).reshape(-1, F)
        weight_grad = np.dot(grad_col.T, x_col.data).reshape(self.weights.shape)
        
        # Add L1 regularization gradient
        if self.l1_lambda > 0:
            weight_grad += self.l1_lambda * np.sign(self.weights)
            
        # Add L2 regularization gradient
        if self.l2_lambda > 0:
            weight_grad += self.l2_lambda * self.weights
            
        self.gradients['weights'] = weight_grad
        
        # Compute input gradient
        w_col = self.weights.reshape(F, -1)
        grad_x_col = np.dot(grad_col, w_col)
        grad_x = Matrix(grad_x_col).col2im(self.input.shape, (KH, KW), self.stride)
        
        return grad_x.data

class MaxPool2D(Layer):
    """2D Max Pooling Layer."""
    
    def __init__(self, pool_size: Union[int, Tuple[int, int]], stride: Optional[int] = None, padding: str = 'valid'):
        """Initialize MaxPool2D layer.
        
        Args:
            pool_size: Size of the pooling window
            stride: Pooling stride (defaults to pool_size)
            padding: Padding mode ('valid' or 'same')
        """
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else pool_size
        self.padding = padding
        self.cache = {}
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input = inputs
        N, C, H, W = inputs.shape
        PH, PW = self.pool_size
        
        # Apply padding if needed
        if self.padding == 'same':
            pad_h = (PH - 1) // 2
            pad_w = (PW - 1) // 2
            inputs = np.pad(inputs, ((0,0), (0,0), (pad_h,pad_h), (pad_w,pad_w)))
            _, _, H, W = inputs.shape
        
        # Calculate output dimensions
        out_h = (H - PH) // self.stride + 1
        out_w = (W - PW) // self.stride + 1
        
        # Reshape input for pooling
        x_reshaped = inputs.reshape(N, C, out_h, self.stride, out_w, self.stride)
        x_col = x_reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, out_h, out_w, -1)
        
        # Perform max pooling
        self.cache['x_col'] = x_col
        max_idx = np.argmax(x_col, axis=4)
        self.cache['max_idx'] = max_idx
        out = np.take_along_axis(x_col, max_idx[..., None], axis=4).squeeze(4)
        
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        N, C, H, W = self.input.shape
        PH, PW = self.pool_size
        
        x_col = self.cache['x_col']
        max_idx = self.cache['max_idx']
        
        # Initialize gradient column matrix
        grad_col = np.zeros_like(x_col)
        grad_flat = grad.reshape(grad.shape[0] * grad.shape[1] * grad.shape[2] * grad.shape[3])
        idx_flat = np.ravel_multi_index(np.indices(max_idx.shape), max_idx.shape)
        
        # Set gradient at maximum indices
        grad_col.reshape(-1)[idx_flat, max_idx.flatten()] = grad_flat
        
        # Reshape gradient to match input shape
        grad_reshaped = grad_col.reshape(N, C, H//PH, W//PW, PH, PW)
        grad_input = grad_reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H, W)
        
        return grad_input

class Dense(Layer):
    """Fully connected layer."""
    
    def __init__(self, units: int, activation: Optional[str] = None,
                 kernel_initializer: str = 'glorot_uniform',
                 l1_lambda: float = 0.0, l2_lambda: float = 0.0):
        """Initialize Dense layer.
        
        Args:
            units: Number of output units
            activation: Activation function name
        """
        super().__init__(l1_lambda, l2_lambda)
        self.units = units
        self.activation = get_activation(activation) if activation else None
        self.kernel_initializer = kernel_initializer
        
        # Initialize parameters
        self.weights = None
        self.bias = None
        self.initialized = False
    
    def initialize(self, input_shape: Tuple[int, ...]):
        """Initialize layer parameters."""
        if not self.initialized:
            input_dim = np.prod(input_shape[1:])
            initializer = get_initializer(self.kernel_initializer)
            self.weights = initializer((input_dim, self.units))
            self.bias = np.zeros(self.units)
            self.initialized = True
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input = inputs
        self.initialize(inputs.shape)
        
        # Reshape input if necessary
        x = inputs.reshape(inputs.shape[0], -1)
        
        # Compute forward pass
        out = np.dot(x, self.weights) + self.bias
        
        if self.activation:
            out = self.activation.forward(out)
        
        self.output = out
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.activation:
            grad = self.activation.backward(self.output) * grad
        
        # Reshape input if necessary
        x = self.input.reshape(self.input.shape[0], -1)
        
        # Compute weights gradient with regularization
        weight_grad = np.dot(x.T, grad)
        
        # Add L1 regularization gradient
        if self.l1_lambda > 0:
            weight_grad += self.l1_lambda * np.sign(self.weights)
            
        # Add L2 regularization gradient
        if self.l2_lambda > 0:
            weight_grad += self.l2_lambda * self.weights
            
        self.gradients['weights'] = weight_grad
        self.gradients['bias'] = np.sum(grad, axis=0)
        
        # Compute input gradient
        grad_input = np.dot(grad, self.weights.T)
        return grad_input.reshape(self.input.shape)


class InceptionModule(Layer):
    """Inception module with parallel convolutional paths."""
    
    def __init__(self, filters_1x1: int, filters_3x3_reduce: int, filters_3x3: int,
                 filters_5x5_reduce: int, filters_5x5: int, filters_pool: int,
                 activation: Optional[str] = None,
                 kernel_initializer: str = 'glorot_uniform'):
        """Initialize Inception module.
        
        Args:
            filters_1x1: Number of filters for 1x1 convolution path
            filters_3x3_reduce: Number of 1x1 filters before 3x3 conv
            filters_3x3: Number of filters for 3x3 convolution path
            filters_5x5_reduce: Number of 1x1 filters before 5x5 conv
            filters_5x5: Number of filters for 5x5 convolution path
            filters_pool: Number of filters for pooling path
            activation: Activation function name
            kernel_initializer: Weight initialization method
        """
        super().__init__()
        
        # 1x1 convolution path
        self.conv1x1 = Conv2D(filters_1x1, kernel_size=1, padding='same',
                             activation=activation, kernel_initializer=kernel_initializer)
        
        # 3x3 convolution path
        self.conv3x3_reduce = Conv2D(filters_3x3_reduce, kernel_size=1, padding='same',
                                    activation=activation, kernel_initializer=kernel_initializer)
        self.conv3x3 = Conv2D(filters_3x3, kernel_size=3, padding='same',
                             activation=activation, kernel_initializer=kernel_initializer)
        
        # 5x5 convolution path
        self.conv5x5_reduce = Conv2D(filters_5x5_reduce, kernel_size=1, padding='same',
                                    activation=activation, kernel_initializer=kernel_initializer)
        self.conv5x5 = Conv2D(filters_5x5, kernel_size=5, padding='same',
                             activation=activation, kernel_initializer=kernel_initializer)
        
        # Pool path
        self.maxpool = MaxPool2D(pool_size=3, stride=1)
        self.conv_pool = Conv2D(filters_pool, kernel_size=1, padding='same',
                               activation=activation, kernel_initializer=kernel_initializer)
    
    def initialize(self, input_shape: Tuple[int, ...]):
        """Initialize all sub-layers."""
        # Initialize all conv layers
        self.conv1x1.initialize(input_shape)
        self.conv3x3_reduce.initialize(input_shape)
        
        # For 3x3 path, the input to conv3x3 has different shape
        reduce_out_shape = (input_shape[0], self.conv3x3_reduce.filters, input_shape[2], input_shape[3])
        self.conv3x3.initialize(reduce_out_shape)
        
        # For 5x5 path
        self.conv5x5_reduce.initialize(input_shape)
        reduce5_out_shape = (input_shape[0], self.conv5x5_reduce.filters, input_shape[2], input_shape[3])
        self.conv5x5.initialize(reduce5_out_shape)
        
        # For pool path
        self.conv_pool.initialize(input_shape)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input = inputs
        
        # Compute each path
        path1 = self.conv1x1.forward(inputs)
        
        path2 = self.conv3x3_reduce.forward(inputs)
        path2 = self.conv3x3.forward(path2)
        
        path3 = self.conv5x5_reduce.forward(inputs)
        path3 = self.conv5x5.forward(path3)
        
        path4 = self.maxpool.forward(inputs)
        path4 = self.conv_pool.forward(path4)
        
        # Concatenate along the channel dimension
        self.output = np.concatenate([path1, path2, path3, path4], axis=1)
        return self.output
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Split gradient for each path
        n1 = self.conv1x1.output.shape[1]
        n2 = self.conv3x3.output.shape[1]
        n3 = self.conv5x5.output.shape[1]
        n4 = self.conv_pool.output.shape[1]
        
        grad1 = grad[:, :n1]
        grad2 = grad[:, n1:n1+n2]
        grad3 = grad[:, n1+n2:n1+n2+n3]
        grad4 = grad[:, n1+n2+n3:]
        
        # Backward pass for each path
        dx1 = self.conv1x1.backward(grad1)
        
        dx2_conv = self.conv3x3.backward(grad2)
        dx2 = self.conv3x3_reduce.backward(dx2_conv)
        
        dx3_conv = self.conv5x5.backward(grad3)
        dx3 = self.conv5x5_reduce.backward(dx3_conv)
        
        dx4_conv = self.conv_pool.backward(grad4)
        dx4 = self.maxpool.backward(dx4_conv)
        
        # Sum gradients from all paths
        return dx1 + dx2 + dx3 + dx4


class ResidualBlock(Layer):
    """Residual block with skip connections."""
    
    def __init__(self, filters: int, kernel_size: int = 3,
                 activation: Optional[str] = 'relu',
                 kernel_initializer: str = 'glorot_uniform',
                 downsample: bool = False):
        """Initialize Residual block.
        
        Args:
            filters: Number of filters for convolutions
            kernel_size: Size of the convolution kernels
            activation: Activation function name
            kernel_initializer: Weight initialization method
            downsample: Whether to downsample spatial dimensions
        """
        super().__init__()
        
        stride = 2 if downsample else 1
        
        # Main path
        self.conv1 = Conv2D(filters, kernel_size=kernel_size,
                           stride=stride, padding='same',
                           activation=activation,
                           kernel_initializer=kernel_initializer)
        
        self.conv2 = Conv2D(filters, kernel_size=kernel_size,
                           stride=1, padding='same',
                           activation=None,
                           kernel_initializer=kernel_initializer)
        
        # Skip connection
        self.shortcut = None
        self.input_filters = None  # Will be set during initialization
        self.downsample = downsample
        self.filters = filters
        
        self.activation = get_activation(activation) if activation else None
    
    def initialize(self, input_shape: Tuple[int, ...]):
        """Initialize layer parameters."""
        self.input_filters = input_shape[1]
        
        # Initialize main convolutions
        self.conv1.initialize(input_shape)
        
        # Calculate shape after conv1
        conv1_shape = (input_shape[0], self.filters, input_shape[2], input_shape[3])
        self.conv2.initialize(conv1_shape)
        
        # Set up shortcut if needed
        if self.downsample or self.input_filters != self.filters:
            stride = 2 if self.downsample else 1
            self.shortcut = Conv2D(self.filters, kernel_size=1,
                                  stride=stride, padding='same',
                                  activation=None)
            self.shortcut.initialize(input_shape)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input = inputs
        self.initialize(inputs.shape)
        
        # Main path
        x = self.conv1.forward(inputs)
        x = self.conv2.forward(x)
        
        # Skip connection
        identity = inputs
        if self.shortcut is not None:
            identity = self.shortcut.forward(inputs)
        
        # Add skip connection
        out = x + identity
        
        if self.activation:
            out = self.activation.forward(out)
        
        self.output = out
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.activation:
            grad = self.activation.backward(self.output) * grad
        
        # Main path gradient
        dx = self.conv2.backward(grad)
        dx = self.conv1.backward(dx)
        
        # Skip connection gradient
        if self.shortcut is not None:
            dx_skip = self.shortcut.backward(grad)
        else:
            dx_skip = grad
        
        return dx + dx_skip


class Dropout(Layer):
    """Dropout layer for regularization."""
    
    def __init__(self, rate: float = 0.5):
        """Initialize Dropout layer.
        
        Args:
            rate: Dropout rate (fraction of units to drop)
        """
        super().__init__()
        self.rate = rate
        self.mask = None
        self.training = True
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input = inputs
        
        if self.training:
            # Generate dropout mask
            self.mask = (np.random.rand(*inputs.shape) > self.rate) / (1 - self.rate)
            return inputs * self.mask
        else:
            return inputs
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask if self.training else grad


class BatchNormalization(Layer):
    """Batch Normalization layer."""
    
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.9):
        """Initialize BatchNormalization layer.
        
        Args:
            epsilon: Small constant for numerical stability
            momentum: Momentum for running statistics
        """
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = None  # Scale parameter
        self.beta = None   # Shift parameter
        self.running_mean = None
        self.running_var = None
        self.cache = {}
        self.training = True
    
    def initialize(self, input_shape: Tuple[int, ...]):
        """Initialize layer parameters."""
        if self.gamma is None:
            self.gamma = np.ones(input_shape[1:])
            self.beta = np.zeros(input_shape[1:])
            self.running_mean = np.zeros(input_shape[1:])
            self.running_var = np.ones(input_shape[1:])
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input = inputs
        self.initialize(inputs.shape)
        
        if self.training:
            # Calculate mean and variance for current batch
            mu = np.mean(inputs, axis=0)
            var = np.var(inputs, axis=0)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # Normalize
            x_norm = (inputs - mu) / np.sqrt(var + self.epsilon)
            
            # Cache values for backward pass
            self.cache.update({
                'mu': mu,
                'var': var,
                'x_norm': x_norm,
            })
        else:
            # Use running statistics for inference
            x_norm = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Scale and shift
        out = self.gamma * x_norm + self.beta
        self.output = out
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Get cached values
        mu = self.cache['mu']
        var = self.cache['var']
        x_norm = self.cache['x_norm']
        
        N = self.input.shape[0]
        x_mu = self.input - mu
        std_inv = 1 / np.sqrt(var + self.epsilon)
        
        # Gradients with respect to gamma and beta
        self.gradients['gamma'] = np.sum(grad * x_norm, axis=0)
        self.gradients['beta'] = np.sum(grad, axis=0)
        
        # Gradient with respect to input
        dx_norm = grad * self.gamma
        dvar = np.sum(dx_norm * x_mu * -0.5 * std_inv**3, axis=0)
        dmu = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2 * x_mu, axis=0)
        
        dx = dx_norm * std_inv + dvar * 2 * x_mu / N + dmu / N
        return dx

class AvgPool2D(Layer):
    """2D Average Pooling Layer."""
    
    def __init__(self, pool_size: Union[int, Tuple[int, int]], stride: Optional[int] = None, padding: str = 'valid'):
        """Initialize AvgPool2D layer.
        
        Args:
            pool_size: Size of the pooling window
            stride: Pooling stride (defaults to pool_size)
            padding: Padding mode ('valid' or 'same')
        """
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else pool_size
        self.padding = padding
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input = inputs
        N, C, H, W = inputs.shape
        PH, PW = self.pool_size
        
        # Calculate output dimensions
        out_h = (H - PH) // self.stride + 1
        out_w = (W - PW) // self.stride + 1
        
        # Initialize output
        output = np.zeros((N, C, out_h, out_w))
        
        # Perform average pooling
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + PH
                w_start = j * self.stride
                w_end = w_start + PW
                
                output[:, :, i, j] = np.mean(inputs[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
        
        self.output = output
        return output
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        N, C, H, W = self.input.shape
        _, _, out_h, out_w = grad.shape
        PH, PW = self.pool_size
        
        # Initialize input gradient
        grad_input = np.zeros_like(self.input)
        
        # Distribute gradient evenly across pooling window
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + PH
                w_start = j * self.stride
                w_end = w_start + PW
                
                grad_input[:, :, h_start:h_end, w_start:w_end] += grad[:, :, i:i+1, j:j+1] / (PH * PW)
        
        return grad_input


class Flatten(Layer):
    """Flatten layer to convert multi-dimensional input to 1D."""
    
    def __init__(self):
        """Initialize Flatten layer."""
        super().__init__()
        self.input_shape = None
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input = inputs
        self.input_shape = inputs.shape
        
        # Flatten all dimensions except batch dimension
        batch_size = inputs.shape[0]
        output = inputs.reshape(batch_size, -1)
        
        self.output = output
        return output
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Reshape gradient back to original input shape
        return grad.reshape(self.input_shape)


class DepthwiseConv2D(Layer):
    """Depthwise separable convolution layer."""
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], 
                 stride: int = 1, padding: str = 'valid',
                 activation: Optional[str] = None,
                 kernel_initializer: str = 'glorot_uniform'):
        """Initialize DepthwiseConv2D layer.
        
        Args:
            kernel_size: Size of the convolution kernel
            stride: Convolution stride
            padding: Padding mode ('valid' or 'same')
            activation: Activation function name
            kernel_initializer: Weight initialization method
        """
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.activation = get_activation(activation) if activation else None
        self.kernel_initializer = kernel_initializer
        
        # Initialize parameters
        self.weights = None
        self.bias = None
        self.initialized = False
    
    def initialize(self, input_shape: Tuple[int, ...]):
        """Initialize layer parameters."""
        if not self.initialized:
            _, channels, _, _ = input_shape
            initializer = get_initializer(self.kernel_initializer)
            # Depthwise: one kernel per input channel
            self.weights = initializer((channels, 1, *self.kernel_size))
            self.bias = np.zeros(channels)
            self.initialized = True
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input = inputs
        self.initialize(inputs.shape)
        
        N, C, H, W = inputs.shape
        KH, KW = self.kernel_size
        
        # Calculate padding
        if self.padding == 'same':
            pad_h = (KH - 1) // 2
            pad_w = (KW - 1) // 2
            inputs = np.pad(inputs, ((0,0), (0,0), (pad_h,pad_h), (pad_w,pad_w)))
            _, _, H, W = inputs.shape
        
        # Calculate output dimensions
        out_h = (H - KH) // self.stride + 1
        out_w = (W - KW) // self.stride + 1
        
        output = np.zeros((N, C, out_h, out_w))
        
        # Apply depthwise convolution
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + KH
                    w_start = j * self.stride
                    w_end = w_start + KW
                    
                    region = inputs[:, c:c+1, h_start:h_end, w_start:w_end]
                    output[:, c, i, j] = np.sum(region * self.weights[c, 0], axis=(1,2,3)) + self.bias[c]
        
        if self.activation:
            output = self.activation.forward(output)
        
        self.output = output
        return output
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.activation:
            grad = self.activation.backward(self.output) * grad
        
        N, C, out_h, out_w = grad.shape
        KH, KW = self.kernel_size
        
        # Compute bias gradient
        self.gradients['bias'] = np.sum(grad, axis=(0, 2, 3))
        
        # Compute weights gradient and input gradient
        self.gradients['weights'] = np.zeros_like(self.weights)
        grad_input = np.zeros_like(self.input)
        
        # Get padded input if needed
        padded_input = self.input
        if self.padding == 'same':
            pad_h = (KH - 1) // 2
            pad_w = (KW - 1) // 2
            padded_input = np.pad(self.input, ((0,0), (0,0), (pad_h,pad_h), (pad_w,pad_w)))
        
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + KH
                    w_start = j * self.stride
                    w_end = w_start + KW
                    
                    region = padded_input[:, c:c+1, h_start:h_end, w_start:w_end]
                    
                    # Weight gradient
                    self.gradients['weights'][c, 0] += np.sum(region * grad[:, c:c+1, i:i+1, j:j+1], axis=0)
                    
                    # Input gradient
                    if self.padding == 'same':
                        if (h_start >= pad_h and h_end <= grad_input.shape[2] + pad_h and
                            w_start >= pad_w and w_end <= grad_input.shape[3] + pad_w):
                            grad_input[:, c, h_start-pad_h:h_end-pad_h, w_start-pad_w:w_end-pad_w] += \
                                self.weights[c, 0] * grad[:, c:c+1, i:i+1, j:j+1]
                    else:
                        grad_input[:, c, h_start:h_end, w_start:w_end] += \
                            self.weights[c, 0] * grad[:, c:c+1, i:i+1, j:j+1]
        
        return grad_input


class Bottleneck(Layer):
    """Bottleneck block for efficient computation."""
    
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int,
                 stride: int = 1, activation: Optional[str] = 'relu',
                 kernel_initializer: str = 'glorot_uniform'):
        """Initialize Bottleneck block.
        
        Args:
            in_channels: Number of input channels
            bottleneck_channels: Number of channels in bottleneck
            out_channels: Number of output channels
            stride: Stride for the middle convolution
            activation: Activation function name
            kernel_initializer: Weight initialization method
        """
        super().__init__()
        
        # 1x1 convolution to reduce channels
        self.conv1 = Conv2D(bottleneck_channels, kernel_size=1,
                           activation=activation, kernel_initializer=kernel_initializer)
        
        # 3x3 convolution with stride
        self.conv2 = Conv2D(bottleneck_channels, kernel_size=3, stride=stride,
                           padding='same', activation=activation,
                           kernel_initializer=kernel_initializer)
        
        # 1x1 convolution to restore channels
        self.conv3 = Conv2D(out_channels, kernel_size=1,
                           activation=None, kernel_initializer=kernel_initializer)
        
        # Skip connection
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2D(out_channels, kernel_size=1, stride=stride,
                                  activation=None, kernel_initializer=kernel_initializer)
        
        self.activation = get_activation(activation) if activation else None
    
    def initialize(self, input_shape: Tuple[int, ...]):
        """Initialize all sub-layers."""
        self.conv1.initialize(input_shape)
        
        # Calculate shape after conv1
        conv1_shape = (input_shape[0], self.conv1.filters, input_shape[2], input_shape[3])
        self.conv2.initialize(conv1_shape)
        
        # Calculate shape after conv2 (may have different spatial dimensions due to stride)
        conv2_h = conv1_shape[2] // self.conv2.stride
        conv2_w = conv1_shape[3] // self.conv2.stride
        conv2_shape = (conv1_shape[0], self.conv2.filters, conv2_h, conv2_w)
        self.conv3.initialize(conv2_shape)
        
        # Initialize shortcut if needed
        if self.shortcut:
            self.shortcut.initialize(input_shape)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input = inputs
        
        # Main path
        out = self.conv1.forward(inputs)
        out = self.conv2.forward(out)
        out = self.conv3.forward(out)
        
        # Skip connection
        shortcut = inputs
        if self.shortcut:
            shortcut = self.shortcut.forward(inputs)
        
        # Add skip connection
        out = out + shortcut
        
        if self.activation:
            out = self.activation.forward(out)
        
        self.output = out
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.activation:
            grad = self.activation.backward(self.output) * grad
        
        # Gradient for main path
        grad_main = self.conv3.backward(grad)
        grad_main = self.conv2.backward(grad_main)
        grad_main = self.conv1.backward(grad_main)
        
        # Gradient for skip connection
        grad_skip = grad
        if self.shortcut:
            grad_skip = self.shortcut.backward(grad)
        
        return grad_main + grad_skip