"""
Unit tests for LSTM implementation.
Tests core functionality, gradient computation, and edge cases.
"""
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.lstm import LSTM, LSTMCell
from src.core.gru import GRU, GRUCell
from src.core.activations import sigmoid, tanh
from src.models.seq2seq import Seq2SeqModel
from src.evaluation.bleu import sentence_bleu, corpus_bleu

def test_lstm_cell():
    """Test basic LSTM cell functionality."""
    print("Testing LSTM Cell...")
    
    batch_size = 4
    input_size = 10
    hidden_size = 8
    
    # Create LSTM cell
    cell = LSTMCell(input_size, hidden_size)
    
    # Test forward pass
    x = np.random.randn(batch_size, input_size)
    h_prev = np.random.randn(batch_size, hidden_size)
    c_prev = np.random.randn(batch_size, hidden_size)
    
    h, c = cell.forward(x, h_prev, c_prev)
    
    # Check output shapes
    assert h.shape == (batch_size, hidden_size), f"Hidden shape mismatch: {h.shape}"
    assert c.shape == (batch_size, hidden_size), f"Cell shape mismatch: {c.shape}"
    
    # Test backward pass
    dh = np.random.randn(batch_size, hidden_size)
    dc = np.random.randn(batch_size, hidden_size)
    
    dx, dh_prev, dc_prev = cell.backward(dh, dc)
    
    # Check gradient shapes
    assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape}"
    assert dh_prev.shape == h_prev.shape, f"dh_prev shape mismatch: {dh_prev.shape}"
    assert dc_prev.shape == c_prev.shape, f"dc_prev shape mismatch: {dc_prev.shape}"
    
    print("✓ LSTM Cell tests passed")

def test_lstm_layer():
    """Test LSTM layer functionality."""
    print("Testing LSTM Layer...")
    
    batch_size = 4
    seq_length = 6
    input_size = 10
    hidden_size = 8
    
    # Create LSTM
    lstm = LSTM(input_size, hidden_size, num_layers=2)
    
    # Test forward pass
    x = np.random.randn(batch_size, seq_length, input_size)
    outputs, final_state = lstm.forward(x)
    
    h_states, c_states = final_state
    
    # Check output shapes
    assert outputs.shape == (batch_size, seq_length, hidden_size)
    assert len(h_states) == 2  # num_layers
    assert len(c_states) == 2  # num_layers
    assert h_states[0].shape == (batch_size, hidden_size)
    assert c_states[0].shape == (batch_size, hidden_size)
    
    print("✓ LSTM Layer tests passed")

def test_bidirectional_lstm():
    """Test bidirectional LSTM."""
    print("Testing Bidirectional LSTM...")
    
    batch_size = 4
    seq_length = 6
    input_size = 10
    hidden_size = 8
    
    # Create bidirectional LSTM
    lstm = LSTM(input_size, hidden_size, bidirectional=True)
    
    # Test forward pass
    x = np.random.randn(batch_size, seq_length, input_size)
    outputs, final_state = lstm.forward(x)
    
    # Check output shapes (should be 2 * hidden_size for bidirectional)
    assert outputs.shape == (batch_size, seq_length, 2 * hidden_size)
    
    print("✓ Bidirectional LSTM tests passed")

def test_gru_cell():
    """Test GRU cell functionality."""
    print("Testing GRU Cell...")
    
    batch_size = 4
    input_size = 10
    hidden_size = 8
    
    # Create GRU cell
    cell = GRUCell(input_size, hidden_size)
    
    # Test forward pass
    x = np.random.randn(batch_size, input_size)
    h_prev = np.random.randn(batch_size, hidden_size)
    
    h = cell.forward(x, h_prev)
    
    # Check output shape
    assert h.shape == (batch_size, hidden_size), f"Hidden shape mismatch: {h.shape}"
    
    # Test backward pass
    dh = np.random.randn(batch_size, hidden_size)
    dx, dh_prev = cell.backward(dh)
    
    # Check gradient shapes
    assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape}"
    assert dh_prev.shape == h_prev.shape, f"dh_prev shape mismatch: {dh_prev.shape}"
    
    print("✓ GRU Cell tests passed")

def test_numerical_gradient():
    """Test gradients using numerical approximation."""
    print("Testing Numerical Gradients...")
    
    def numerical_gradient(f, x, h=1e-5):
        """Compute numerical gradient."""
        grad = np.zeros_like(x)
        flat_x = x.flatten()
        flat_grad = grad.flatten()
        
        for i in range(len(flat_x)):
            old_value = flat_x[i]
            
            flat_x[i] = old_value + h
            x_plus = flat_x.reshape(x.shape)
            fxh_plus = f(x_plus)
            
            flat_x[i] = old_value - h
            x_minus = flat_x.reshape(x.shape)
            fxh_minus = f(x_minus)
            
            flat_grad[i] = (fxh_plus - fxh_minus) / (2 * h)
            flat_x[i] = old_value
        
        return grad
    
    # Test simple function: f(x) = sum(tanh(x))
    x = np.random.randn(3, 4) * 0.1  # Small values for numerical stability
    
    def f(x):
        return np.sum(tanh(x))
    
    analytical_grad = 1 - tanh(x) ** 2  # Derivative of tanh
    numerical_grad = numerical_gradient(f, x)
    
    # Check if gradients are close
    diff = np.abs(analytical_grad - numerical_grad)
    max_diff = np.max(diff)
    
    assert max_diff < 1e-4, f"Gradient mismatch: max diff = {max_diff}"
    
    print(f"✓ Numerical gradient test passed (max diff: {max_diff:.2e})")

def test_seq2seq_model():
    """Test Seq2Seq model functionality."""
    print("Testing Seq2Seq Model...")
    
    src_vocab_size = 100
    tgt_vocab_size = 80
    batch_size = 4
    src_seq_length = 10
    tgt_seq_length = 8
    
    # Create model
    model = Seq2SeqModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embedding_dim=32,
        hidden_size=64,
        num_layers=1,
        attention_type='bahdanau'
    )
    
    # Test forward pass
    src_ids = np.random.randint(0, src_vocab_size, (batch_size, src_seq_length))
    tgt_ids = np.random.randint(0, tgt_vocab_size, (batch_size, tgt_seq_length))
    
    logits, attention_weights = model.forward(src_ids, tgt_ids)
    
    # Check output shapes
    assert logits.shape == (batch_size, tgt_seq_length, tgt_vocab_size)
    if attention_weights is not None:
        assert attention_weights.shape == (batch_size, tgt_seq_length, src_seq_length)
    
    # Test translation
    translations, _ = model.translate(src_ids, max_length=15)
    assert translations.shape[0] == batch_size
    
    print("✓ Seq2Seq Model tests passed")

def test_bleu_score():
    """Test BLEU score computation."""
    print("Testing BLEU Score...")
    
    # Test cases
    test_cases = [
        {
            'candidate': 'the cat is on the mat',
            'references': ['the cat is on the mat'],
            'expected_range': (0.9, 1.0)
        },
        {
            'candidate': 'the cat is on the mat',
            'references': ['the cat is sitting on the mat'],
            'expected_range': (0.5, 0.9)
        },
        {
            'candidate': 'hello world',
            'references': ['goodbye world'],
            'expected_range': (0.0, 0.5)
        }
    ]
    
    for i, case in enumerate(test_cases):
        score = sentence_bleu(case['candidate'], case['references'])
        min_score, max_score = case['expected_range']
        
        assert min_score <= score <= max_score, \
            f"Test case {i}: BLEU score {score} not in range {case['expected_range']}"
    
    # Test corpus BLEU
    candidates = ['the cat is on the mat', 'hello world']
    references_list = [['the cat is on the mat'], ['hello world']]
    
    corpus_score = corpus_bleu(candidates, references_list)
    assert 0.0 <= corpus_score <= 1.0, f"Corpus BLEU score out of range: {corpus_score}"
    
    print("✓ BLEU Score tests passed")

def test_edge_cases():
    """Test edge cases and error conditions."""
    print("Testing Edge Cases...")
    
    # Test with very small batch size
    lstm = LSTM(input_size=5, hidden_size=4)
    x = np.random.randn(1, 3, 5)  # batch_size=1
    outputs, _ = lstm.forward(x)
    assert outputs.shape == (1, 3, 4)
    
    # Test with single time step
    x = np.random.randn(2, 1, 5)  # seq_length=1
    outputs, _ = lstm.forward(x)
    assert outputs.shape == (2, 1, 4)
    
    # Test with zero dropout
    lstm_no_dropout = LSTM(input_size=5, hidden_size=4, dropout_rate=0.0)
    x = np.random.randn(2, 3, 5)
    outputs1, _ = lstm_no_dropout.forward(x, training=True)
    outputs2, _ = lstm_no_dropout.forward(x, training=True)
    
    # Outputs should be identical with no dropout
    assert np.allclose(outputs1, outputs2, rtol=1e-5)
    
    print("✓ Edge case tests passed")

def test_memory_consistency():
    """Test that forward and backward passes don't corrupt memory."""
    print("Testing Memory Consistency...")
    
    lstm = LSTM(input_size=8, hidden_size=6, num_layers=2)
    
    # Store initial parameters
    initial_params = [p.copy() for p in lstm.get_parameters()]
    
    # Multiple forward passes
    for _ in range(5):
        x = np.random.randn(3, 4, 8)
        outputs, _ = lstm.forward(x, training=True)
    
    # Check that parameters haven't changed (no backward pass called)
    current_params = lstm.get_parameters()
    
    for initial, current in zip(initial_params, current_params):
        assert np.allclose(initial, current), "Parameters changed without backward pass"
    
    print("✓ Memory consistency tests passed")

def run_all_tests():
    """Run all unit tests."""
    print("Running LSTM Implementation Tests")
    print("=" * 40)
    
    try:
        test_lstm_cell()
        test_lstm_layer()
        test_bidirectional_lstm()
        test_gru_cell()
        test_numerical_gradient()
        test_seq2seq_model()
        test_bleu_score()
        test_edge_cases()
        test_memory_consistency()
        
        print("\n" + "=" * 40)
        print("✅ All tests passed successfully!")
        print("=" * 40)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        raise

if __name__ == "__main__":
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    run_all_tests() 