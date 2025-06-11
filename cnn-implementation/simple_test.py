#!/usr/bin/env python3
"""
Simple CNN Test
===============

Basic test to verify core functionality works.
"""

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_cnn():
    """Test basic CNN functionality."""
    print("Testing Basic CNN Functionality")
    print("=" * 40)
    
    try:
        # Import core components
        from core.layers.base import Conv2D, MaxPool2D, Dense, Flatten
        from core.model import Sequential
        from core.activations import ReLU
        from core.optimizers import Adam
        from core.losses import CrossEntropy
        
        print("✅ All imports successful")
        
        # Create a simple model
        model = Sequential()
        model.add(Conv2D(16, 3, padding='same', activation='relu'))
        model.add(MaxPool2D(2))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))
        
        print("✅ Model created successfully")
        
        # Test forward pass
        x = np.random.randn(2, 3, 8, 8)
        y = model.forward(x)
        
        print(f"✅ Forward pass: {x.shape} -> {y.shape}")
        
        # Test individual layers
        conv = Conv2D(8, 3, padding='same', activation='relu')
        conv.initialize(x.shape)
        conv_out = conv.forward(x)
        print(f"✅ Conv2D: {x.shape} -> {conv_out.shape}")
        
        pool = MaxPool2D(2)
        pool_out = pool.forward(conv_out)
        print(f"✅ MaxPool2D: {conv_out.shape} -> {pool_out.shape}")
        
        flatten = Flatten()
        flat_out = flatten.forward(pool_out)
        print(f"✅ Flatten: {pool_out.shape} -> {flat_out.shape}")
        
        dense = Dense(5, activation='relu')
        dense.initialize(flat_out.shape)
        dense_out = dense.forward(flat_out)
        print(f"✅ Dense: {flat_out.shape} -> {dense_out.shape}")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimizers():
    """Test optimizer functionality."""
    print("\nTesting Optimizers")
    print("=" * 40)
    
    try:
        from core.optimizers import SGD, Adam, RMSprop
        
        # Test with dummy parameters
        params = {'weights': np.random.randn(3, 3), 'bias': np.random.randn(3)}
        grads = {'weights': np.random.randn(3, 3), 'bias': np.random.randn(3)}
        
        # Test SGD
        sgd = SGD(learning_rate=0.01, momentum=0.9)
        sgd.update(params.copy(), grads)
        print("✅ SGD optimizer")
        
        # Test Adam
        adam = Adam(learning_rate=0.001)
        adam.update(params.copy(), grads)
        print("✅ Adam optimizer")
        
        # Test RMSprop
        rmsprop = RMSprop(learning_rate=0.01)
        rmsprop.update(params.copy(), grads)
        print("✅ RMSprop optimizer")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimizer test failed: {e}")
        return False

def test_activations():
    """Test activation functions."""
    print("\nTesting Activations")
    print("=" * 40)
    
    try:
        from core.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
        
        x = np.array([-2, -1, 0, 1, 2])
        
        # Test ReLU
        relu = ReLU()
        relu_out = relu.forward(x)
        print(f"✅ ReLU: {x} -> {relu_out}")
        
        # Test LeakyReLU
        leaky = LeakyReLU(alpha=0.01)
        leaky_out = leaky.forward(x)
        print(f"✅ LeakyReLU: {x} -> {leaky_out}")
        
        # Test Sigmoid
        sigmoid = Sigmoid()
        sig_out = sigmoid.forward(x)
        print(f"✅ Sigmoid: {x} -> {sig_out}")
        
        return True
        
    except Exception as e:
        print(f"❌ Activation test failed: {e}")
        return False

def test_matrix_operations():
    """Test matrix operations."""
    print("\nTesting Matrix Operations")
    print("=" * 40)
    
    try:
        from core.matrix import Matrix
        
        # Test basic operations
        m1 = Matrix(np.random.randn(3, 3))
        m2 = Matrix(np.random.randn(3, 3))
        
        # Test addition
        m3 = m1 + m2
        print(f"✅ Matrix addition: {m1.shape} + {m2.shape} = {m3.shape}")
        
        # Test multiplication
        m4 = m1 * 2.0
        print(f"✅ Scalar multiplication: {m1.shape} * 2.0 = {m4.shape}")
        
        # Test transpose
        m5 = m1.T
        print(f"✅ Transpose: {m1.shape} -> {m5.shape}")
        
        # Test im2col
        x = np.random.randn(1, 3, 4, 4)
        matrix = Matrix(x)
        x_col = matrix.im2col((3, 3), stride=1)
        print(f"✅ im2col: {x.shape} -> {x_col.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Matrix test failed: {e}")
        return False

if __name__ == "__main__":
    print("CNN Implementation Simple Test")
    print("=" * 50)
    
    tests = [
        test_matrix_operations,
        test_activations,
        test_optimizers,
        test_basic_cnn
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS: {passed}/{total} tests passed")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Basic CNN implementation is working.")
    else:
        print("⚠️  Some tests failed. Check implementation.") 