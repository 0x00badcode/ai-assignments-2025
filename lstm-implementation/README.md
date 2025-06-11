# LSTM Implementation from Scratch

A comprehensive implementation of LSTM (Long Short-Term Memory) networks from scratch using NumPy, designed for neural machine translation and sequence-to-sequence learning.

## Features

### Core Implementation
- ✅ LSTM cell implementation from scratch using NumPy
- ✅ Vectorized forward and backward pass optimization
- ✅ Multiple activation functions (tanh, leaky_relu, elu, sigmoid, relu)
- ✅ Recurrent dropout for regularization
- ✅ GRU implementation for comparison

### Architectures
- ✅ Sequence-to-Sequence (Seq2Seq) for Neural Machine Translation
- ✅ Encoder-Decoder architecture
- ✅ Bahdanau Attention mechanism
- ✅ Luong Attention mechanism

### Evaluation & Optimization
- ✅ BLEU score evaluation for translation quality
- ✅ Training optimizations (Adam optimizer, gradient clipping)
- ✅ Comparison with TensorFlow/PyTorch implementations

## Project Structure

```
lstm-implementation/
├── src/
│   ├── core/
│   │   ├── lstm.py           # Core LSTM implementation
│   │   ├── gru.py            # GRU implementation
│   │   ├── activations.py    # Activation functions
│   │   └── utils.py          # Utility functions
│   ├── models/
│   │   ├── seq2seq.py        # Seq2Seq model
│   │   ├── encoder.py        # Encoder implementation
│   │   ├── decoder.py        # Decoder implementation
│   │   └── attention.py      # Attention mechanisms
│   ├── training/
│   │   ├── optimizer.py      # Optimizers (Adam, SGD)
│   │   ├── trainer.py        # Training loop
│   │   └── losses.py         # Loss functions
│   └── evaluation/
│       ├── bleu.py           # BLEU score implementation
│       └── metrics.py        # Other evaluation metrics
├── examples/
│   ├── translation.py        # NMT example
│   └── comparison.py         # TensorFlow/PyTorch comparison
├── data/
│   └── preprocessing.py      # Data preprocessing utilities
├── tests/
│   └── test_lstm.py          # Unit tests
└── requirements.txt          # Dependencies

```

## Usage

### Basic LSTM Usage
```python
from src.core.lstm import LSTM
from src.core.activations import tanh

# Create LSTM layer
lstm = LSTM(input_size=100, hidden_size=128, activation=tanh)

# Forward pass
hidden, cell = lstm.forward(inputs, initial_hidden, initial_cell)
```

### Seq2Seq for Translation
```python
from src.models.seq2seq import Seq2SeqModel

# Create model
model = Seq2SeqModel(
    vocab_size_src=10000,
    vocab_size_tgt=8000,
    embedding_dim=256,
    hidden_dim=512,
    attention_type='bahdanau'
)

# Train model
model.train(train_data, epochs=100)

# Translate
translation = model.translate("Hello world")
```

## References
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.2329)
- [Machine Learning Mastery - Dropout in LSTM](https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/) 