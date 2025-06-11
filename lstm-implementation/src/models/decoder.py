"""
Decoder implementation for Seq2Seq models with attention support.
"""
import numpy as np
from ..core.lstm import LSTM
from ..core.activations import softmax
from ..core.utils import xavier_uniform
from .attention import create_attention

class Decoder:
    """
    LSTM-based decoder with attention mechanism for sequence-to-sequence models.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, encoder_hidden_size=None,
                 num_layers=1, dropout_rate=0.0, recurrent_dropout_rate=0.0, 
                 attention_type=None, **attention_kwargs):
        """
        Initialize decoder.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Hidden size of LSTM
            encoder_hidden_size: Hidden size of encoder (for attention)
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for LSTM
            recurrent_dropout_rate: Recurrent dropout rate for LSTM
            attention_type: Type of attention mechanism ('bahdanau', 'luong', None)
            **attention_kwargs: Additional arguments for attention mechanism
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size or hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.attention_type = attention_type
        
        # Initialize embedding layer
        self.embedding = xavier_uniform((vocab_size, embedding_dim))
        
        # Determine LSTM input size
        lstm_input_size = embedding_dim
        if attention_type is not None:
            lstm_input_size += self.encoder_hidden_size  # Add context vector
        
        # Initialize LSTM
        self.lstm = LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            recurrent_dropout_rate=recurrent_dropout_rate,
            bidirectional=False  # Decoder is always unidirectional
        )
        
        # Initialize attention mechanism
        self.attention = None
        if attention_type is not None:
            self.attention = create_attention(
                attention_type, self.encoder_hidden_size, hidden_size, **attention_kwargs
            )
        
        # Output projection layer
        output_input_size = hidden_size
        if attention_type is not None:
            output_input_size += self.encoder_hidden_size  # Add context vector
        
        self.output_projection = xavier_uniform((output_input_size, vocab_size))
        self.output_bias = np.zeros((vocab_size,))
        
        # Cache for backward pass
        self.cache = {}
    
    def forward_step(self, input_id, hidden_state, cell_state, encoder_outputs=None, 
                    encoder_mask=None, training=True):
        """
        Forward pass for a single time step.
        
        Args:
            input_id: Input token ID (batch_size,)
            hidden_state: Previous hidden state (batch_size, hidden_size)
            cell_state: Previous cell state (batch_size, hidden_size)
            encoder_outputs: Encoder outputs for attention (batch_size, seq_len, encoder_hidden_size)
            encoder_mask: Mask for encoder outputs (batch_size, seq_len)
            training: Whether in training mode
        
        Returns:
            output_logits: Output logits (batch_size, vocab_size)
            new_hidden_state: New hidden state (batch_size, hidden_size)
            new_cell_state: New cell state (batch_size, hidden_size)
            attention_weights: Attention weights (batch_size, seq_len) or None
        """
        batch_size = input_id.shape[0]
        
        # Embedding lookup
        embedding = self.embedding[input_id]  # (batch_size, embedding_dim)
        
        # Attention mechanism
        context_vector = None
        attention_weights = None
        if self.attention is not None and encoder_outputs is not None:
            context_vector, attention_weights = self.attention.forward(
                encoder_outputs, hidden_state, encoder_mask
            )
            # Concatenate embedding with context vector
            lstm_input = np.concatenate([embedding, context_vector], axis=1)
        else:
            lstm_input = embedding
        
        # LSTM forward pass
        lstm_input_expanded = lstm_input[:, np.newaxis, :]  # Add time dimension
        lstm_outputs, (new_hidden_states, new_cell_states) = self.lstm.forward(
            lstm_input_expanded, 
            initial_state=([hidden_state], [cell_state]),
            training=training
        )
        
        new_hidden_state = new_hidden_states[0]
        new_cell_state = new_cell_states[0]
        lstm_output = lstm_outputs[:, 0, :]  # Remove time dimension
        
        # Output projection
        if context_vector is not None:
            projection_input = np.concatenate([lstm_output, context_vector], axis=1)
        else:
            projection_input = lstm_output
        
        output_logits = np.dot(projection_input, self.output_projection) + self.output_bias
        
        return output_logits, new_hidden_state, new_cell_state, attention_weights
    
    def forward(self, target_ids, encoder_outputs, encoder_final_state, 
               encoder_mask=None, training=True):
        """
        Forward pass through decoder (teacher forcing during training).
        
        Args:
            target_ids: Target token IDs (batch_size, target_seq_length)
            encoder_outputs: Encoder outputs (batch_size, src_seq_length, encoder_hidden_size)
            encoder_final_state: Final encoder state (h, c)
            encoder_mask: Mask for encoder outputs (batch_size, src_seq_length)
            training: Whether in training mode
        
        Returns:
            output_logits: Output logits (batch_size, target_seq_length, vocab_size)
            attention_weights: Attention weights (batch_size, target_seq_length, src_seq_length) or None
        """
        batch_size, target_seq_length = target_ids.shape
        
        # Initialize decoder state from encoder final state
        if isinstance(encoder_final_state, tuple):
            h_states, c_states = encoder_final_state
            if isinstance(h_states, list):
                # Multiple layers or bidirectional
                hidden_state = h_states[0]  # Use first layer/direction
                cell_state = c_states[0]
            else:
                hidden_state = h_states
                cell_state = c_states
        else:
            # GRU case (no cell state)
            hidden_state = encoder_final_state[0] if isinstance(encoder_final_state, list) else encoder_final_state
            cell_state = np.zeros_like(hidden_state)
        
        # Ensure correct dimensions
        if hidden_state.shape[1] != self.hidden_size:
            # Project encoder hidden state to decoder hidden size
            if not hasattr(self, 'state_projection'):
                self.state_projection = xavier_uniform((hidden_state.shape[1], self.hidden_size))
            hidden_state = np.dot(hidden_state, self.state_projection)
            cell_state = np.dot(cell_state, self.state_projection)
        
        outputs = []
        attention_weights_list = []
        
        # Process each time step (teacher forcing)
        for t in range(target_seq_length):
            input_id = target_ids[:, t]  # Current target token
            
            output_logits, hidden_state, cell_state, attention_weights = self.forward_step(
                input_id, hidden_state, cell_state, encoder_outputs, encoder_mask, training
            )
            
            outputs.append(output_logits)
            if attention_weights is not None:
                attention_weights_list.append(attention_weights)
        
        # Stack outputs
        output_logits = np.stack(outputs, axis=1)  # (batch_size, target_seq_length, vocab_size)
        
        if attention_weights_list:
            attention_weights = np.stack(attention_weights_list, axis=1)  # (batch_size, target_seq_length, src_seq_length)
        else:
            attention_weights = None
        
        # Cache for backward pass
        self.cache = {
            'target_ids': target_ids,
            'encoder_outputs': encoder_outputs,
            'encoder_final_state': encoder_final_state,
            'encoder_mask': encoder_mask,
            'output_logits': output_logits,
            'attention_weights': attention_weights
        }
        
        return output_logits, attention_weights
    
    def generate(self, encoder_outputs, encoder_final_state, encoder_mask=None,
                max_length=50, start_token=1, end_token=2, temperature=1.0):
        """
        Generate sequence using greedy decoding or sampling.
        
        Args:
            encoder_outputs: Encoder outputs (batch_size, src_seq_length, encoder_hidden_size)
            encoder_final_state: Final encoder state
            encoder_mask: Mask for encoder outputs (batch_size, src_seq_length)
            max_length: Maximum generation length
            start_token: Start token ID
            end_token: End token ID
            temperature: Sampling temperature (1.0 = no scaling)
        
        Returns:
            generated_ids: Generated token IDs (batch_size, generated_length)
            attention_weights: Attention weights (batch_size, generated_length, src_seq_length) or None
        """
        batch_size = encoder_outputs.shape[0]
        
        # Initialize decoder state
        if isinstance(encoder_final_state, tuple):
            h_states, c_states = encoder_final_state
            if isinstance(h_states, list):
                hidden_state = h_states[0]
                cell_state = c_states[0]
            else:
                hidden_state = h_states
                cell_state = c_states
        else:
            hidden_state = encoder_final_state[0] if isinstance(encoder_final_state, list) else encoder_final_state
            cell_state = np.zeros_like(hidden_state)
        
        # Project if necessary
        if hidden_state.shape[1] != self.hidden_size:
            hidden_state = np.dot(hidden_state, self.state_projection)
            cell_state = np.dot(cell_state, self.state_projection)
        
        # Initialize with start token
        current_input = np.full((batch_size,), start_token, dtype=np.int32)
        
        generated_tokens = []
        attention_weights_list = []
        finished = np.zeros((batch_size,), dtype=bool)
        
        for t in range(max_length):
            # Forward step
            output_logits, hidden_state, cell_state, attention_weights = self.forward_step(
                current_input, hidden_state, cell_state, encoder_outputs, encoder_mask, training=False
            )
            
            # Apply temperature scaling
            if temperature != 1.0:
                output_logits = output_logits / temperature
            
            # Sample next token (greedy for now)
            next_token = np.argmax(output_logits, axis=1)
            
            # Update finished sequences
            finished = finished | (next_token == end_token)
            
            # Store results
            generated_tokens.append(next_token)
            if attention_weights is not None:
                attention_weights_list.append(attention_weights)
            
            # Check if all sequences are finished
            if np.all(finished):
                break
            
            # Update input for next step
            current_input = next_token
        
        # Stack results
        generated_ids = np.stack(generated_tokens, axis=1)  # (batch_size, generated_length)
        
        if attention_weights_list:
            attention_weights = np.stack(attention_weights_list, axis=1)
        else:
            attention_weights = None
        
        return generated_ids, attention_weights
    
    def backward(self, dlogits, dattention_weights=None):
        """
        Backward pass through decoder.
        
        Args:
            dlogits: Gradient w.r.t. output logits (batch_size, target_seq_length, vocab_size)
            dattention_weights: Gradient w.r.t. attention weights (optional)
        
        Returns:
            dencoder_outputs: Gradient w.r.t. encoder outputs
        """
        # Implementation would be quite complex for full backward pass
        # For now, we'll implement a simplified version
        cache = self.cache
        target_ids = cache['target_ids']
        encoder_outputs = cache['encoder_outputs']
        
        # Gradients w.r.t. output projection
        batch_size, target_seq_length, vocab_size = dlogits.shape
        
        # Backward through output projection
        dprojection_input = np.dot(dlogits.reshape(-1, vocab_size), self.output_projection.T)
        self.doutput_projection = np.dot(dprojection_input.T, dlogits.reshape(-1, vocab_size))
        self.doutput_bias = np.sum(dlogits, axis=(0, 1))
        
        # Simplified: return zero gradients for encoder outputs
        dencoder_outputs = np.zeros_like(encoder_outputs)
        
        return dencoder_outputs
    
    def get_parameters(self):
        """Get all trainable parameters."""
        params = [self.embedding, self.output_projection, self.output_bias]
        params.extend(self.lstm.get_parameters())
        
        if self.attention is not None:
            if hasattr(self.attention, 'W_a'):
                params.append(self.attention.W_a)
            if hasattr(self.attention, 'U_a'):
                params.append(self.attention.U_a)
            if hasattr(self.attention, 'v_a'):
                params.append(self.attention.v_a)
            if hasattr(self.attention, 'b_a'):
                params.append(self.attention.b_a)
        
        if hasattr(self, 'state_projection'):
            params.append(self.state_projection)
        
        return params
    
    def get_gradients(self):
        """Get all gradients."""
        grads = [getattr(self, 'dembedding', np.zeros_like(self.embedding)),
                self.doutput_projection, self.doutput_bias]
        grads.extend(self.lstm.get_gradients())
        
        if self.attention is not None:
            if hasattr(self.attention, 'dW_a'):
                grads.append(self.attention.dW_a)
            if hasattr(self.attention, 'dU_a'):
                grads.append(self.attention.dU_a)
            if hasattr(self.attention, 'dv_a'):
                grads.append(self.attention.dv_a)
            if hasattr(self.attention, 'db_a'):
                grads.append(self.attention.db_a)
        
        if hasattr(self, 'dstate_projection'):
            grads.append(self.dstate_projection)
        
        return grads 