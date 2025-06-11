"""
Encoder implementation for Seq2Seq models.
"""
import numpy as np
from ..core.lstm import LSTM
from ..core.utils import xavier_uniform

class Encoder:
    """
    LSTM-based encoder for sequence-to-sequence models.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, 
                 dropout_rate=0.0, recurrent_dropout_rate=0.0, bidirectional=False):
        """
        Initialize encoder.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Hidden size of LSTM
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for LSTM
            recurrent_dropout_rate: Recurrent dropout rate for LSTM
            bidirectional: Whether to use bidirectional LSTM
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.bidirectional = bidirectional
        
        # Initialize embedding layer
        self.embedding = xavier_uniform((vocab_size, embedding_dim))
        
        # Initialize LSTM
        self.lstm = LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            recurrent_dropout_rate=recurrent_dropout_rate,
            bidirectional=bidirectional
        )
        
        # Cache for backward pass
        self.cache = {}
    
    def forward(self, input_ids, input_mask=None, training=True):
        """
        Forward pass through encoder.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_length)
            input_mask: Mask for padding tokens (batch_size, seq_length)
            training: Whether in training mode
        
        Returns:
            outputs: Encoder outputs (batch_size, seq_length, hidden_size * num_directions)
            final_state: Final LSTM states (h, c)
        """
        batch_size, seq_length = input_ids.shape
        
        # Embedding lookup
        embeddings = self.embedding[input_ids]  # (batch_size, seq_length, embedding_dim)
        
        # Apply input mask to embeddings if provided
        if input_mask is not None:
            mask_expanded = input_mask[:, :, np.newaxis]  # (batch_size, seq_length, 1)
            embeddings = embeddings * mask_expanded
        
        # Pass through LSTM
        outputs, final_state = self.lstm.forward(embeddings, training=training)
        
        # Apply output mask if provided
        if input_mask is not None:
            num_directions = 2 if self.bidirectional else 1
            mask_expanded = input_mask[:, :, np.newaxis]  # (batch_size, seq_length, 1)
            mask_expanded = np.broadcast_to(mask_expanded, (batch_size, seq_length, self.hidden_size * num_directions))
            outputs = outputs * mask_expanded
        
        # Cache for backward pass
        self.cache = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'embeddings': embeddings,
            'outputs': outputs,
            'final_state': final_state
        }
        
        return outputs, final_state
    
    def backward(self, doutputs, dfinal_state=None):
        """
        Backward pass through encoder.
        
        Args:
            doutputs: Gradient w.r.t. outputs (batch_size, seq_length, hidden_size * num_directions)
            dfinal_state: Gradient w.r.t. final state
        
        Returns:
            dinput_ids: Gradient w.r.t. input IDs (not used, but for completeness)
        """
        # Retrieve cached values
        cache = self.cache
        input_ids, input_mask = cache['input_ids'], cache['input_mask']
        embeddings = cache['embeddings']
        
        # Apply output mask to gradients if provided
        if input_mask is not None:
            num_directions = 2 if self.bidirectional else 1
            mask_expanded = input_mask[:, :, np.newaxis]
            mask_expanded = np.broadcast_to(mask_expanded, doutputs.shape)
            doutputs = doutputs * mask_expanded
        
        # Backward through LSTM
        dembeddings = self.lstm.backward(doutputs, dfinal_state)
        
        # Apply input mask to embedding gradients if provided
        if input_mask is not None:
            mask_expanded = input_mask[:, :, np.newaxis]
            mask_expanded = np.broadcast_to(mask_expanded, dembeddings.shape)
            dembeddings = dembeddings * mask_expanded
        
        # Backward through embedding layer
        self.dembedding = np.zeros_like(self.embedding)
        for i in range(input_ids.shape[0]):
            for j in range(input_ids.shape[1]):
                token_id = input_ids[i, j]
                self.dembedding[token_id] += dembeddings[i, j]
        
        return None  # No gradient w.r.t. discrete input IDs
    
    def get_parameters(self):
        """Get all trainable parameters."""
        params = [self.embedding]
        params.extend(self.lstm.get_parameters())
        return params
    
    def get_gradients(self):
        """Get all gradients."""
        grads = [self.dembedding]
        grads.extend(self.lstm.get_gradients())
        return grads

class BiLSTMEncoder(Encoder):
    """
    Bidirectional LSTM encoder with separate forward and backward processing.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, 
                 dropout_rate=0.0, recurrent_dropout_rate=0.0):
        """
        Initialize bidirectional encoder.
        """
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            recurrent_dropout_rate=recurrent_dropout_rate,
            bidirectional=True
        )
    
    def forward(self, input_ids, input_mask=None, training=True):
        """
        Forward pass with explicit bidirectional processing.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_length)
            input_mask: Mask for padding tokens (batch_size, seq_length)
            training: Whether in training mode
        
        Returns:
            outputs: Concatenated forward and backward outputs (batch_size, seq_length, 2 * hidden_size)
            final_state: Final states from both directions
        """
        batch_size, seq_length = input_ids.shape
        
        # Embedding lookup
        embeddings = self.embedding[input_ids]  # (batch_size, seq_length, embedding_dim)
        
        # Apply input mask to embeddings if provided
        if input_mask is not None:
            mask_expanded = input_mask[:, :, np.newaxis]
            embeddings = embeddings * mask_expanded
        
        # Forward direction
        forward_outputs = []
        forward_h = np.zeros((batch_size, self.hidden_size))
        forward_c = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_length):
            forward_h, forward_c = self.lstm.cells[0][0].forward(
                embeddings[:, t, :], forward_h, forward_c, training
            )
            forward_outputs.append(forward_h)
        
        # Backward direction
        backward_outputs = []
        backward_h = np.zeros((batch_size, self.hidden_size))
        backward_c = np.zeros((batch_size, self.hidden_size))
        
        for t in reversed(range(seq_length)):
            backward_h, backward_c = self.lstm.cells[0][1].forward(
                embeddings[:, t, :], backward_h, backward_c, training
            )
            backward_outputs.append(backward_h)
        
        # Reverse backward outputs to match time order
        backward_outputs = list(reversed(backward_outputs))
        
        # Concatenate forward and backward outputs
        outputs = []
        for t in range(seq_length):
            combined = np.concatenate([forward_outputs[t], backward_outputs[t]], axis=1)
            outputs.append(combined)
        
        outputs = np.stack(outputs, axis=1)  # (batch_size, seq_length, 2 * hidden_size)
        
        # Apply output mask if provided
        if input_mask is not None:
            mask_expanded = input_mask[:, :, np.newaxis]
            mask_expanded = np.broadcast_to(mask_expanded, outputs.shape)
            outputs = outputs * mask_expanded
        
        # Final states
        final_state = ([forward_h, backward_h], [forward_c, backward_c])
        
        # Cache for backward pass
        self.cache = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'embeddings': embeddings,
            'outputs': outputs,
            'final_state': final_state,
            'forward_outputs': forward_outputs,
            'backward_outputs': backward_outputs
        }
        
        return outputs, final_state

def create_encoder(encoder_type, **kwargs):
    """
    Factory function to create encoder.
    
    Args:
        encoder_type: Type of encoder ('lstm', 'bilstm')
        **kwargs: Arguments for encoder initialization
    
    Returns:
        Encoder instance
    """
    if encoder_type.lower() == 'lstm':
        return Encoder(**kwargs)
    elif encoder_type.lower() == 'bilstm':
        return BiLSTMEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}") 