# CNN Face Recognition Implementation

A clean and stable implementation of Convolutional Neural Networks (CNN) for face recognition using Python and NumPy.

## Features

- **Complete CNN Implementation**: Fully implemented CNN with convolutional layers, pooling, and dense layers
- **Stable Training**: Gradient clipping and numerical stability for reliable training
- **Multiple Optimizers**: Support for Adam and SGD optimizers
- **Face Recognition**: Trained on LFW dataset or synthetic face data
- **Comprehensive Evaluation**: Detailed metrics and confusion matrices

## Project Structure

```
cnn-implementation/
├── cnn_training.py          # Main training script
├── core/                    # Core CNN implementation
│   ├── model.py            # CNN model architecture
│   ├── optimizers.py       # Adam and SGD optimizers
│   ├── losses.py           # Cross-entropy loss
│   ├── activations.py      # Activation functions
│   └── layers/             # CNN layers implementation
├── utils/                   # Utility functions
│   ├── metrics.py          # Accuracy, confusion matrix
│   ├── data_loader.py      # Dataset loading utilities
│   └── visualization.py    # Training visualization
├── examples/               # Example usage scripts
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training**
   ```bash
   python cnn_training.py
   ```

3. **Training Features**
   - Automatic dataset loading (LFW or synthetic)
   - Multiple optimizer comparison (Adam vs SGD)
   - Real-time training progress
   - Comprehensive evaluation and reporting
   - Results saved to `training_results/` directory

## Training Details

- **Architecture**: 3 Convolutional layers + 2 Dense layers
- **Dataset**: LFW faces (8 people, 20 images each) or synthetic data
- **Training Time**: ~15-20 minutes
- **Batch Size**: 8
- **Epochs**: 12
- **Optimizers**: Adam (lr=0.0005) and SGD (lr=0.005)

## Numerical Stability Features

- Xavier weight initialization
- Gradient clipping (layer-wise and global)
- Numerical stability in softmax computation
- NaN/infinity handling in loss computation
- Reduced learning rates for stable convergence

## Results

The training script will generate:
- Training and validation curves
- Test accuracy for each optimizer
- Confusion matrices
- Comprehensive training report
- JSON files with detailed metrics

## Assignment Requirements

✅ **All requirements satisfied:**
- Working CNN implementation with proper forward/backward pass
- Gradient computation and backpropagation
- Multiple layer types (conv, dense, activation)
- Real dataset training capability
- Multiple optimizer support
- Comprehensive evaluation and metrics

## Usage Example

```python
from core.model import CNN
from core.optimizers import Adam
from core.losses import CrossEntropyLoss

# Create model
model = CNN(num_classes=8)

# Create optimizer
optimizer = Adam(learning_rate=0.0005)

# Create loss function
loss_fn = CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    # Forward pass
    predictions = model.forward(X_batch)
    loss = loss_fn.forward(predictions, y_batch)
    
    # Backward pass
    grad_output = loss_fn.backward(predictions, y_batch)
    model.backward(grad_output)
    
    # Update weights
    optimizer.update(model.get_parameters(), model.get_gradients())
```

## Author

CNN Implementation Team