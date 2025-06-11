# CNN Implementation from Scratch

This project implements a Convolutional Neural Network (CNN) framework from scratch, without relying on deep learning libraries like PyTorch or TensorFlow. The implementation focuses on flexibility, performance, and educational value.

## Features

### Core Components
- Matrix operations and computational graphs
- Various layer types (Convolution, Pooling, Fully Connected, etc.)
- Multiple activation functions (ReLU, LeakyReLU, etc.)
- Different optimization algorithms (SGD, Momentum, RMSProp, Adam)

### Advanced Features
- Customizable network architecture
- Multiple weight initialization methods
- Regularization techniques (L1, L2, Elastic Net)
- Modern architectural blocks (Inception, Residual)
- Optimized convolution operations

### Training & Evaluation
- Support for classification and regression tasks
- Training progress monitoring
- Performance metrics and visualization
- Model saving and loading

## Project Structure

```
cnn_implementation/
├── core/                   # Core implementations
│   ├── layers/            # Layer implementations
│   ├── activations.py     # Activation functions
│   ├── initializers.py    # Weight initialization methods
│   ├── losses.py          # Loss functions
│   └── optimizers.py      # Optimization algorithms
├── utils/                 # Utility functions
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── metrics.py         # Evaluation metrics
│   └── visualization.py   # Training visualization
├── models/                # Model implementations
│   ├── base.py           # Base model class
│   └── blocks.py         # Advanced architectural blocks
└── examples/             # Usage examples and demos
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cnn-implementation.git
cd cnn-implementation

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from cnn_implementation.core.layers import Conv2D, MaxPool2D, Dense
from cnn_implementation.core.activations import ReLU
from cnn_implementation.models import Sequential

# Define model architecture
model = Sequential([
    Conv2D(filters=32, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=2),
    Conv2D(filters=64, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=2),
    Dense(units=128, activation='relu'),
    Dense(units=10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate model
model.evaluate(x_test, y_test)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.