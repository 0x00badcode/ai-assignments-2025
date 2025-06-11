"""
GRU (Gated Recurrent Unit) implementation from scratch using NumPy.
For comparison with LSTM implementation.
"""
import numpy as np
from .activations import sigmoid, tanh, sigmoid_derivative, tanh_derivative
from .utils import xavier_uniform, orthogonal_init, recurrent_dropout_mask

class GRUCell:
    """Single GRU cell implementation."""
    
    def __init__(self, input_size, hidden_size, dropout_rate=0.0, recurrent_dropout_rate=0.0):
        """
        Initialize GRU cell.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            dropout_rate: Dropout rate for input connections
            recurrent_dropout_rate: Dropout rate for recurrent connections
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        
        # Initialize weights
        self._init_weights()
        
        # Cache for backward pass
        self.cache = {}
    
    def _init_weights(self):
        """Initialize GRU weights using Xavier initialization."""
        # Input weights for reset and update gates (input_size -> 2 * hidden_size)
        self.W_ir = xavier_uniform((self.input_size, self.hidden_size))  # Reset gate
        self.W_iz = xavier_uniform((self.input_size, self.hidden_size))  # Update gate
        self.W_ih = xavier_uniform((self.input_size, self.hidden_size))  # New gate
        
        # Recurrent weights for reset and update gates (hidden_size -> 2 * hidden_size)
        self.W_hr = orthogonal_init((self.hidden_size, self.hidden_size))  # Reset gate
        self.W_hz = orthogonal_init((self.hidden_size, self.hidden_size))  # Update gate
        self.W_hh = orthogonal_init((self.hidden_size, self.hidden_size))  # New gate
        
        # Biases
        self.b_ir = np.zeros((self.hidden_size,))
        self.b_iz = np.zeros((self.hidden_size,))
        self.b_ih = np.zeros((self.hidden_size,))
        self.b_hr = np.zeros((self.hidden_size,))
        self.b_hz = np.zeros((self.hidden_size,))
        self.b_hh = np.zeros((self.hidden_size,))
    
    def forward(self, x, h_prev, training=True):
        """
        Forward pass through GRU cell.
        
        Args:
            x: Input at current time step (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
            training: Whether in training mode
        
        Returns:
            h: New hidden state (batch_size, hidden_size)
        """
        batch_size = x.shape[0]
        
        # Generate dropout masks
        input_mask = recurrent_dropout_mask(
            (batch_size, self.input_size), 
            self.dropout_rate, 
            training
        )
        recurrent_mask = recurrent_dropout_mask(
            (batch_size, self.hidden_size), 
            self.recurrent_dropout_rate, 
            training
        )
        
        # Apply dropout
        x_dropped = x * input_mask
        h_dropped = h_prev * recurrent_mask
        
        # Reset gate
        r = sigmoid(np.dot(x_dropped, self.W_ir) + self.b_ir + 
                   np.dot(h_dropped, self.W_hr) + self.b_hr)
        
        # Update gate
        z = sigmoid(np.dot(x_dropped, self.W_iz) + self.b_iz + 
                   np.dot(h_dropped, self.W_hz) + self.b_hz)
        
        # New gate (candidate hidden state)
        h_tilde = tanh(np.dot(x_dropped, self.W_ih) + self.b_ih + 
                      np.dot(r * h_dropped, self.W_hh) + self.b_hh)
        
        # Update hidden state
        h = (1 - z) * h_prev + z * h_tilde
        
        # Cache for backward pass
        self.cache = {
            'x': x, 'h_prev': h_prev,
            'x_dropped': x_dropped, 'h_dropped': h_dropped,
            'r': r, 'z': z, 'h_tilde': h_tilde, 'h': h,
            'input_mask': input_mask, 'recurrent_mask': recurrent_mask
        }
        
        return h
    
    def backward(self, dh):
        """
        Backward pass through GRU cell.
        
        Args:
            dh: Gradient w.r.t. hidden state (batch_size, hidden_size)
        
        Returns:
            dx: Gradient w.r.t. input (batch_size, input_size)
            dh_prev: Gradient w.r.t. previous hidden state (batch_size, hidden_size)
        """
        # Retrieve cached values
        cache = self.cache
        x, h_prev = cache['x'], cache['h_prev']
        x_dropped, h_dropped = cache['x_dropped'], cache['h_dropped']
        r, z, h_tilde, h = cache['r'], cache['z'], cache['h_tilde'], cache['h']
        input_mask, recurrent_mask = cache['input_mask'], cache['recurrent_mask']
        
        # Gradient w.r.t. update gate
        dz = dh * (h_tilde - h_prev)
        dz_raw = dz * sigmoid_derivative(z)
        
        # Gradient w.r.t. candidate hidden state
        dh_tilde = dh * z
        dh_tilde_raw = dh_tilde * tanh_derivative(h_tilde)
        
        # Gradient w.r.t. reset gate
        dr = np.dot(dh_tilde_raw, self.W_hh.T) * h_dropped
        dr_raw = dr * sigmoid_derivative(r)
        
        # Gradients w.r.t. weights and biases
        self.dW_ir = np.dot(x_dropped.T, dr_raw)
        self.dW_iz = np.dot(x_dropped.T, dz_raw)
        self.dW_ih = np.dot(x_dropped.T, dh_tilde_raw)
        
        self.dW_hr = np.dot(h_dropped.T, dr_raw)
        self.dW_hz = np.dot(h_dropped.T, dz_raw)
        self.dW_hh = np.dot((r * h_dropped).T, dh_tilde_raw)
        
        self.db_ir = np.sum(dr_raw, axis=0)
        self.db_iz = np.sum(dz_raw, axis=0)
        self.db_ih = np.sum(dh_tilde_raw, axis=0)
        self.db_hr = np.sum(dr_raw, axis=0)
        self.db_hz = np.sum(dz_raw, axis=0)
        self.db_hh = np.sum(dh_tilde_raw, axis=0)
        
        # Gradients w.r.t. inputs
        dx_dropped = (np.dot(dr_raw, self.W_ir.T) + 
                     np.dot(dz_raw, self.W_iz.T) + 
                     np.dot(dh_tilde_raw, self.W_ih.T))
        
        dh_dropped = (np.dot(dr_raw, self.W_hr.T) + 
                     np.dot(dz_raw, self.W_hz.T) + 
                     np.dot(dh_tilde_raw * r, self.W_hh.T))
        
        # Apply dropout masks
        dx = dx_dropped * input_mask
        dh_prev = dh_dropped * recurrent_mask + dh * (1 - z)
        
        return dx, dh_prev

class GRU:
    """Multi-layer GRU implementation."""
    
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 dropout_rate=0.0, recurrent_dropout_rate=0.0, bidirectional=False):
        """
        Initialize GRU.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            num_layers: Number of GRU layers
            dropout_rate: Dropout rate for input connections
            recurrent_dropout_rate: Dropout rate for recurrent connections
            bidirectional: Whether to use bidirectional GRU
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.bidirectional = bidirectional
        
        # Create GRU cells
        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * (2 if bidirectional else 1)
            
            # Forward cells
            forward_cell = GRUCell(
                layer_input_size, hidden_size, 
                dropout_rate, recurrent_dropout_rate
            )
            
            if bidirectional:
                # Backward cells
                backward_cell = GRUCell(
                    layer_input_size, hidden_size,
                    dropout_rate, recurrent_dropout_rate
                )
                self.cells.append((forward_cell, backward_cell))
            else:
                self.cells.append(forward_cell)
    
    def forward(self, x, initial_state=None, training=True):
        """
        Forward pass through GRU.
        
        Args:
            x: Input sequences (batch_size, seq_length, input_size)
            initial_state: Initial hidden states or None
            training: Whether in training mode
        
        Returns:
            outputs: Output sequences (batch_size, seq_length, hidden_size * num_directions)
            final_state: Final hidden states
        """
        batch_size, seq_length, _ = x.shape
        num_directions = 2 if self.bidirectional else 1
        
        # Initialize states
        if initial_state is None:
            h_states = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers * num_directions)]
        else:
            h_states = initial_state
        
        # Store outputs for each time step
        outputs = []
        
        # Process each time step
        for t in range(seq_length):
            layer_input = x[:, t, :]
            
            # Process each layer
            for layer in range(self.num_layers):
                if self.bidirectional:
                    forward_cell, backward_cell = self.cells[layer]
                    
                    # Forward direction
                    h_f = forward_cell.forward(layer_input, h_states[layer*2], training)
                    h_states[layer*2] = h_f
                    
                    # Backward direction (process in reverse)
                    h_b = backward_cell.forward(layer_input, h_states[layer*2+1], training)
                    h_states[layer*2+1] = h_b
                    
                    # Concatenate forward and backward outputs
                    layer_output = np.concatenate([h_f, h_b], axis=1)
                else:
                    cell = self.cells[layer]
                    h = cell.forward(layer_input, h_states[layer], training)
                    h_states[layer] = h
                    layer_output = h
                
                layer_input = layer_output
            
            outputs.append(layer_output)
        
        # Stack outputs
        outputs = np.stack(outputs, axis=1)  # (batch_size, seq_length, hidden_size * num_directions)
        
        return outputs, h_states
    
    def get_parameters(self):
        """Get all trainable parameters."""
        params = []
        if self.bidirectional:
            for forward_cell, backward_cell in self.cells:
                params.extend([
                    forward_cell.W_ir, forward_cell.W_iz, forward_cell.W_ih,
                    forward_cell.W_hr, forward_cell.W_hz, forward_cell.W_hh,
                    forward_cell.b_ir, forward_cell.b_iz, forward_cell.b_ih,
                    forward_cell.b_hr, forward_cell.b_hz, forward_cell.b_hh
                ])
                params.extend([
                    backward_cell.W_ir, backward_cell.W_iz, backward_cell.W_ih,
                    backward_cell.W_hr, backward_cell.W_hz, backward_cell.W_hh,
                    backward_cell.b_ir, backward_cell.b_iz, backward_cell.b_ih,
                    backward_cell.b_hr, backward_cell.b_hz, backward_cell.b_hh
                ])
        else:
            for cell in self.cells:
                params.extend([
                    cell.W_ir, cell.W_iz, cell.W_ih,
                    cell.W_hr, cell.W_hz, cell.W_hh,
                    cell.b_ir, cell.b_iz, cell.b_ih,
                    cell.b_hr, cell.b_hz, cell.b_hh
                ])
        return params
    
    def get_gradients(self):
        """Get all gradients."""
        grads = []
        if self.bidirectional:
            for forward_cell, backward_cell in self.cells:
                grads.extend([
                    forward_cell.dW_ir, forward_cell.dW_iz, forward_cell.dW_ih,
                    forward_cell.dW_hr, forward_cell.dW_hz, forward_cell.dW_hh,
                    forward_cell.db_ir, forward_cell.db_iz, forward_cell.db_ih,
                    forward_cell.db_hr, forward_cell.db_hz, forward_cell.db_hh
                ])
                grads.extend([
                    backward_cell.dW_ir, backward_cell.dW_iz, backward_cell.dW_ih,
                    backward_cell.dW_hr, backward_cell.dW_hz, backward_cell.dW_hh,
                    backward_cell.db_ir, backward_cell.db_iz, backward_cell.db_ih,
                    backward_cell.db_hr, backward_cell.db_hz, backward_cell.db_hh
                ])
        else:
            for cell in self.cells:
                grads.extend([
                    cell.dW_ir, cell.dW_iz, cell.dW_ih,
                    cell.dW_hr, cell.dW_hz, cell.dW_hh,
                    cell.db_ir, cell.db_iz, cell.db_ih,
                    cell.db_hr, cell.db_hz, cell.db_hh
                ])
        return grads 