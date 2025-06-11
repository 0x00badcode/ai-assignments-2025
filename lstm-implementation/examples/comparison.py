"""
Comparison of custom LSTM implementation with TensorFlow and PyTorch.
Benchmarks performance, accuracy, and training speed.
"""
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.lstm import LSTM
from src.core.gru import GRU
from src.models.seq2seq import Seq2SeqModel

# Try importing TensorFlow and PyTorch
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    print("TensorFlow version:", tf.__version__)
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    print("PyTorch version:", torch.__version__)
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available")

def create_test_data(batch_size=32, seq_length=20, input_size=50, num_samples=1000):
    """Create synthetic test data for benchmarking."""
    np.random.seed(42)
    X = np.random.randn(num_samples, seq_length, input_size).astype(np.float32)
    y = np.random.randint(0, 2, (num_samples, seq_length)).astype(np.int32)
    return X, y

def benchmark_custom_lstm():
    """Benchmark custom LSTM implementation."""
    print("\n" + "="*50)
    print("Benchmarking Custom LSTM Implementation")
    print("="*50)
    
    # Parameters
    batch_size = 32
    seq_length = 20
    input_size = 50
    hidden_size = 64
    num_samples = 1000
    
    # Create test data
    X, y = create_test_data(batch_size, seq_length, input_size, num_samples)
    
    # Create custom LSTM
    lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2)
    
    # Warm-up run
    outputs, _ = lstm.forward(X[:batch_size], training=True)
    
    # Benchmark forward pass
    start_time = time.time()
    num_batches = num_samples // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_x = X[start_idx:end_idx]
        
        outputs, _ = lstm.forward(batch_x, training=True)
    
    forward_time = time.time() - start_time
    
    print(f"Custom LSTM Performance:")
    print(f"  Input shape: {X[:batch_size].shape}")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Forward pass time: {forward_time:.4f}s for {num_batches} batches")
    print(f"  Time per batch: {forward_time/num_batches:.4f}s")
    print(f"  Parameters: {sum(p.size for p in lstm.get_parameters())}")
    
    return forward_time, outputs.shape

def benchmark_tensorflow_lstm():
    """Benchmark TensorFlow LSTM implementation."""
    if not TF_AVAILABLE:
        print("\nTensorFlow not available, skipping benchmark.")
        return None, None
    
    print("\n" + "="*50)
    print("Benchmarking TensorFlow LSTM Implementation")
    print("="*50)
    
    # Parameters
    batch_size = 32
    seq_length = 20
    input_size = 50
    hidden_size = 64
    num_samples = 1000
    
    # Create test data
    X, y = create_test_data(batch_size, seq_length, input_size, num_samples)
    
    # Create TensorFlow LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1),
        tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
    ])
    
    # Build model
    model.build(input_shape=(None, seq_length, input_size))
    
    # Warm-up run
    outputs = model(X[:batch_size], training=True)
    
    # Benchmark forward pass
    start_time = time.time()
    num_batches = num_samples // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_x = X[start_idx:end_idx]
        
        outputs = model(batch_x, training=True)
    
    forward_time = time.time() - start_time
    
    print(f"TensorFlow LSTM Performance:")
    print(f"  Input shape: {X[:batch_size].shape}")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Forward pass time: {forward_time:.4f}s for {num_batches} batches")
    print(f"  Time per batch: {forward_time/num_batches:.4f}s")
    print(f"  Parameters: {model.count_params()}")
    
    return forward_time, outputs.shape

def benchmark_pytorch_lstm():
    """Benchmark PyTorch LSTM implementation."""
    if not TORCH_AVAILABLE:
        print("\nPyTorch not available, skipping benchmark.")
        return None, None
    
    print("\n" + "="*50)
    print("Benchmarking PyTorch LSTM Implementation")
    print("="*50)
    
    # Parameters
    batch_size = 32
    seq_length = 20
    input_size = 50
    hidden_size = 64
    num_samples = 1000
    
    # Create test data
    X, y = create_test_data(batch_size, seq_length, input_size, num_samples)
    X_torch = torch.from_numpy(X)
    
    # Create PyTorch LSTM
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=0.1)
        
        def forward(self, x):
            output, _ = self.lstm(x)
            return output
    
    model = LSTMModel(input_size, hidden_size)
    model.train()
    
    # Warm-up run
    with torch.no_grad():
        outputs = model(X_torch[:batch_size])
    
    # Benchmark forward pass
    start_time = time.time()
    num_batches = num_samples // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = X_torch[start_idx:end_idx]
            
            outputs = model(batch_x)
    
    forward_time = time.time() - start_time
    
    print(f"PyTorch LSTM Performance:")
    print(f"  Input shape: {X_torch[:batch_size].shape}")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Forward pass time: {forward_time:.4f}s for {num_batches} batches")
    print(f"  Time per batch: {forward_time/num_batches:.4f}s")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params}")
    
    return forward_time, outputs.shape

def compare_lstm_vs_gru():
    """Compare LSTM vs GRU performance."""
    print("\n" + "="*50)
    print("Comparing LSTM vs GRU Performance")
    print("="*50)
    
    # Parameters
    batch_size = 32
    seq_length = 20
    input_size = 50
    hidden_size = 64
    num_samples = 1000
    
    # Create test data
    X, y = create_test_data(batch_size, seq_length, input_size, num_samples)
    
    # Create models
    lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2)
    gru = GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2)
    
    # Benchmark LSTM
    start_time = time.time()
    num_batches = num_samples // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_x = X[start_idx:end_idx]
        
        outputs, _ = lstm.forward(batch_x, training=True)
    
    lstm_time = time.time() - start_time
    
    # Benchmark GRU
    start_time = time.time()
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_x = X[start_idx:end_idx]
        
        outputs, _ = gru.forward(batch_x, training=True)
    
    gru_time = time.time() - start_time
    
    print(f"LSTM vs GRU Comparison:")
    print(f"  LSTM time: {lstm_time:.4f}s")
    print(f"  GRU time: {gru_time:.4f}s")
    print(f"  Speedup (GRU vs LSTM): {lstm_time/gru_time:.2f}x")
    print(f"  LSTM parameters: {sum(p.size for p in lstm.get_parameters())}")
    print(f"  GRU parameters: {sum(p.size for p in gru.get_parameters())}")

def test_numerical_stability():
    """Test numerical stability of implementations."""
    print("\n" + "="*50)
    print("Testing Numerical Stability")
    print("="*50)
    
    # Parameters
    batch_size = 16
    seq_length = 50
    input_size = 100
    hidden_size = 128
    
    # Create test data with extreme values
    X_normal = np.random.randn(batch_size, seq_length, input_size).astype(np.float32)
    X_large = 10 * X_normal  # Large values
    X_small = 0.001 * X_normal  # Small values
    
    lstm = LSTM(input_size=input_size, hidden_size=hidden_size)
    
    print("Testing with different input magnitudes:")
    
    for name, X in [("Normal", X_normal), ("Large", X_large), ("Small", X_small)]:
        try:
            outputs, _ = lstm.forward(X, training=True)
            
            # Check for NaN or Inf
            has_nan = np.isnan(outputs).any()
            has_inf = np.isinf(outputs).any()
            
            print(f"  {name:6s}: Mean={np.mean(outputs):.6f}, "
                  f"Std={np.std(outputs):.6f}, "
                  f"NaN={has_nan}, Inf={has_inf}")
            
        except Exception as e:
            print(f"  {name:6s}: Error - {str(e)}")

def accuracy_comparison():
    """Compare accuracy on a simple task."""
    print("\n" + "="*50)
    print("Accuracy Comparison on Sequence Classification")
    print("="*50)
    
    # Create simple sequence classification task
    np.random.seed(42)
    
    batch_size = 32
    seq_length = 10
    input_size = 1
    num_samples = 1000
    
    # Generate data: classify if sequence sum > 0
    X = np.random.randn(num_samples, seq_length, input_size).astype(np.float32)
    y = (np.sum(X.squeeze(), axis=1) > 0).astype(np.int32)
    
    print(f"Task: Binary classification based on sequence sum")
    print(f"Data: {X.shape}, Labels: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # For simplicity, we'll just test that our models can process the data
    # A full training comparison would require more implementation
    
    lstm = LSTM(input_size=input_size, hidden_size=32)
    gru = GRU(input_size=input_size, hidden_size=32)
    
    # Test forward pass
    batch_x = X[:batch_size]
    
    lstm_out, _ = lstm.forward(batch_x, training=True)
    gru_out, _ = gru.forward(batch_x, training=True)
    
    print(f"LSTM output shape: {lstm_out.shape}")
    print(f"GRU output shape: {gru_out.shape}")
    print("Both models successfully processed the data.")

def run_all_benchmarks():
    """Run all benchmarks and comparisons."""
    print("LSTM Implementation Comparison and Benchmarks")
    print("=" * 60)
    
    # Store results
    results = {}
    
    # Custom implementation
    custom_time, custom_shape = benchmark_custom_lstm()
    results['custom'] = {'time': custom_time, 'shape': custom_shape}
    
    # TensorFlow
    tf_time, tf_shape = benchmark_tensorflow_lstm()
    if tf_time is not None:
        results['tensorflow'] = {'time': tf_time, 'shape': tf_shape}
    
    # PyTorch
    torch_time, torch_shape = benchmark_pytorch_lstm()
    if torch_time is not None:
        results['pytorch'] = {'time': torch_time, 'shape': torch_shape}
    
    # LSTM vs GRU
    compare_lstm_vs_gru()
    
    # Numerical stability
    test_numerical_stability()
    
    # Accuracy comparison
    accuracy_comparison()
    
    # Summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    if len(results) > 1:
        custom_time = results['custom']['time']
        
        print("Relative Performance (Custom LSTM = 1.0x):")
        for framework, data in results.items():
            if framework != 'custom':
                speedup = custom_time / data['time']
                print(f"  {framework.capitalize():12s}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    
    print("\nNote: Performance can vary significantly based on:")
    print("  - Hardware (CPU vs GPU)")
    print("  - Optimization level")
    print("  - Memory layout")
    print("  - BLAS libraries")
    print("  - Batch size and sequence length")

if __name__ == "__main__":
    run_all_benchmarks() 