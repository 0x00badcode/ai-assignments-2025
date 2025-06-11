import numpy as np
from typing import Union, Tuple, List

class Matrix:
    """A class for efficient matrix operations optimized for CNN computations."""
    
    def __init__(self, data: Union[np.ndarray, List]):
        """Initialize Matrix object with numpy array or list.
        
        Args:
            data: Input data as numpy array or list
        """
        self.data = np.array(data, dtype=np.float32)
        
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the matrix."""
        return self.data.shape
    
    @property
    def T(self) -> 'Matrix':
        """Get the transpose of the matrix."""
        return Matrix(self.data.T)
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Matrix addition."""
        return Matrix(self.data + other.data)
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """Matrix subtraction."""
        return Matrix(self.data - other.data)
    
    def __mul__(self, other: Union['Matrix', float, int]) -> 'Matrix':
        """Element-wise multiplication."""
        if isinstance(other, (float, int)):
            return Matrix(self.data * other)
        return Matrix(self.data * other.data)
    
    def __rmul__(self, other: Union[float, int]) -> 'Matrix':
        """Reverse multiplication for scalar * Matrix."""
        if isinstance(other, (float, int)):
            return Matrix(other * self.data)
        return NotImplemented
    
    def dot(self, other: 'Matrix') -> 'Matrix':
        """Matrix multiplication."""
        return Matrix(np.dot(self.data, other.data))
    
    def transpose(self) -> 'Matrix':
        """Matrix transpose."""
        return Matrix(self.data.T)
    
    def reshape(self, *shape: int) -> 'Matrix':
        """Reshape matrix to new dimensions."""
        return Matrix(self.data.reshape(shape))
    
    def pad(self, pad_width: Union[int, Tuple[int, ...]], mode: str = 'constant') -> 'Matrix':
        """Pad matrix with zeros or other values.
        
        Args:
            pad_width: Number of values padded to each edge
            mode: Padding mode ('constant', 'reflect', etc.)
        """
        return Matrix(np.pad(self.data, pad_width, mode=mode))
    
    @staticmethod
    def zeros(*shape: int) -> 'Matrix':
        """Create matrix of zeros with given shape."""
        return Matrix(np.zeros(shape))
    
    @staticmethod
    def ones(*shape: int) -> 'Matrix':
        """Create matrix of ones with given shape."""
        return Matrix(np.ones(shape))
    
    @staticmethod
    def random(*shape: int, mean: float = 0.0, std: float = 1.0) -> 'Matrix':
        """Create matrix of random values with given shape.
        
        Args:
            shape: Dimensions of the matrix
            mean: Mean of the normal distribution
            std: Standard deviation of the normal distribution
        """
        return Matrix(np.random.normal(mean, std, shape))
    
    def im2col(self, kernel_size: Tuple[int, int], stride: int = 1) -> 'Matrix':
        """Convert image matrix to column matrix for efficient convolution.
        
        Args:
            kernel_size: Size of the convolution kernel (height, width)
            stride: Convolution stride
            
        Returns:
            Matrix containing all possible windows of size kernel_size
        """
        # Get dimensions
        N, C, H, W = self.data.shape
        kh, kw = kernel_size
        
        # Calculate output dimensions
        out_h = (H - kh) // stride + 1
        out_w = (W - kw) // stride + 1
        
        # Create strided view of input
        stride_shape = (N, C, out_h, out_w, kh, kw)
        strides = (self.data.strides[0], self.data.strides[1],
                  stride * self.data.strides[2], stride * self.data.strides[3],
                  self.data.strides[2], self.data.strides[3])
        
        # Get all possible windows
        windows = np.lib.stride_tricks.as_strided(
            self.data, shape=stride_shape, strides=strides, writeable=False
        )
        
        # Reshape to 2D matrix
        cols = windows.reshape(N * out_h * out_w, C * kh * kw)
        return Matrix(cols)
    
    def col2im(self, output_shape: Tuple[int, ...], kernel_size: Tuple[int, int], 
               stride: int = 1, padding: int = 0) -> 'Matrix':
        """Convert column matrix back to image format (reverse of im2col).
        
        Args:
            output_shape: Expected output shape (N, C, H, W)
            kernel_size: Size of the kernel (KH, KW)
            stride: Stride of the convolution
            padding: Padding applied
            
        Returns:
            Matrix with image format
        """
        N, C, H, W = output_shape
        KH, KW = kernel_size
        
        # Add padding to output dimensions
        H_padded = H + 2 * padding
        W_padded = W + 2 * padding
        
        # Initialize output
        img = np.zeros((N, C, H_padded, W_padded))
        
        # Calculate dimensions after convolution
        out_h = (H_padded - KH) // stride + 1
        out_w = (W_padded - KW) // stride + 1
        
        # Reshape column data
        col = self.data.reshape(N, out_h, out_w, C, KH, KW)
        
        # Reconstruct image
        for y in range(KH):
            y_max = y + stride * out_h
            for x in range(KW):
                x_max = x + stride * out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, :, :, y, x].transpose(0, 3, 1, 2)
        
        # Remove padding if applied
        if padding > 0:
            img = img[:, :, padding:-padding, padding:-padding]
        
        return Matrix(img)
    
    def fft_convolve2d(self, kernel: 'Matrix') -> 'Matrix':
        """Perform 2D convolution using FFT for efficiency.
        
        Args:
            kernel: Convolution kernel
            
        Returns:
            Convolved matrix
        """
        try:
            import scipy.signal
            result = scipy.signal.fftconvolve(self.data, kernel.data, mode='valid')
            return Matrix(result)
        except ImportError:
            # Fallback to manual FFT implementation
            return self._manual_fft_convolve(kernel)
    
    def _manual_fft_convolve(self, kernel: 'Matrix') -> 'Matrix':
        """Manual FFT convolution implementation.
        
        Args:
            kernel: Convolution kernel
            
        Returns:
            Convolved matrix
        """
        # Get dimensions
        img_h, img_w = self.data.shape[-2:]
        ker_h, ker_w = kernel.data.shape[-2:]
        
        # Calculate output size
        out_h = img_h - ker_h + 1
        out_w = img_w - ker_w + 1
        
        # Pad for FFT
        fft_h = img_h + ker_h - 1
        fft_w = img_w + ker_w - 1
        
        # Pad images
        img_padded = np.pad(self.data, 
                           ((0, 0), (0, 0), (0, fft_h - img_h), (0, fft_w - img_w)))
        ker_padded = np.pad(kernel.data,
                           ((0, 0), (0, 0), (0, fft_h - ker_h), (0, fft_w - ker_w)))
        
        # Perform FFT convolution
        img_fft = np.fft.fft2(img_padded, axes=(-2, -1))
        ker_fft = np.fft.fft2(ker_padded, axes=(-2, -1))
        
        # Convolution in frequency domain
        conv_fft = img_fft * ker_fft
        
        # Convert back to spatial domain
        conv_result = np.fft.ifft2(conv_fft, axes=(-2, -1)).real
        
        # Extract valid region
        valid_result = conv_result[..., :out_h, :out_w]
        
        return Matrix(valid_result)