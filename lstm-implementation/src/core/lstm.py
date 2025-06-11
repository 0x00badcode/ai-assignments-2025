"""
LSTM (Long Short-Term Memory) implementation from scratch using NumPy.
Includes vectorized forward and backward passes with recurrent dropout support.
"""
import numpy as np
from .activations import sigmoid, tanh, sigmoid_derivative, tanh_derivative
from .utils import xavier_uniform, orthogonal_init, recurrent_dropout_mask

class LSTMCell:
    """Single LSTM cell implementation."""
    
    def __init__(self, input_size, hidden_size, dropout_rate=0.0, recurrent_dropout_rate=0.0):
        """
        Initialize LSTM cell.
        
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
        """Initialize LSTM weights using Xavier initialization."""
        # Input weights (input_size -> 4 * hidden_size)
        self.W_i = xavier_uniform((self.input_size, 4 * self.hidden_size))
        
        # Recurrent weights (hidden_size -> 4 * hidden_size)
        self.W_h = orthogonal_init((self.hidden_size, 4 * self.hidden_size))
        
        # Biases
        self.b = np.zeros((4 * self.hidden_size,))
        
        # Initialize forget gate bias to 1 (indices hidden_size:2*hidden_size)
        self.b[self.hidden_size:2*self.hidden_size] = 1.0
    
    def forward(self, x, h_prev, c_prev, training=True):
        """
        Forward pass through LSTM cell.
        
        Args:
            x: Input at current time step (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
            c_prev: Previous cell state (batch_size, hidden_size)
            training: Whether in training mode
        
        Returns:
            h: New hidden state (batch_size, hidden_size)
            c: New cell state (batch_size, hidden_size)
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
        
        # Compute gates (vectorized)
        # i: input gate, f: forget gate, g: candidate values, o: output gate
        gates = np.dot(x_dropped, self.W_i) + np.dot(h_dropped, self.W_h) + self.b
        
        # Split gates
        i = sigmoid(gates[:, :self.hidden_size])                    # Input gate
        f = sigmoid(gates[:, self.hidden_size:2*self.hidden_size])  # Forget gate
        g = tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])   # Candidate values
        o = sigmoid(gates[:, 3*self.hidden_size:])                  # Output gate
        
        # Update cell state
        c = f * c_prev + i * g
        
        # Update hidden state
        h = o * tanh(c)
        
        # Cache for backward pass
        self.cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'x_dropped': x_dropped, 'h_dropped': h_dropped,
            'gates': gates, 'i': i, 'f': f, 'g': g, 'o': o,
            'c': c, 'h': h,
            'input_mask': input_mask, 'recurrent_mask': recurrent_mask
        }
        
        return h, c
    
    def backward(self, dh, dc):
        """
        Backward pass through LSTM cell.
        
        Args:
            dh: Gradient w.r.t. hidden state (batch_size, hidden_size)
            dc: Gradient w.r.t. cell state (batch_size, hidden_size)
        
        Returns:
            dx: Gradient w.r.t. input (batch_size, input_size)
            dh_prev: Gradient w.r.t. previous hidden state (batch_size, hidden_size)
            dc_prev: Gradient w.r.t. previous cell state (batch_size, hidden_size)
        """
        # Retrieve cached values
        cache = self.cache
        x, h_prev, c_prev = cache['x'], cache['h_prev'], cache['c_prev']
        x_dropped, h_dropped = cache['x_dropped'], cache['h_dropped']
        gates, i, f, g, o = cache['gates'], cache['i'], cache['f'], cache['g'], cache['o']
        c, h = cache['c'], cache['h']
        input_mask, recurrent_mask = cache['input_mask'], cache['recurrent_mask']
        
        # Gradient w.r.t. output gate
        do = dh * tanh(c)
        do_raw = do * sigmoid_derivative(gates[:, 3*self.hidden_size:])
        
        # Gradient w.r.t. cell state
        dc = dc + dh * o * tanh_derivative(c)
        
        # Gradient w.r.t. forget gate
        df = dc * c_prev
        df_raw = df * sigmoid_derivative(gates[:, self.hidden_size:2*self.hidden_size])
        
        # Gradient w.r.t. input gate
        di = dc * g
        di_raw = di * sigmoid_derivative(gates[:, :self.hidden_size])
        
        # Gradient w.r.t. candidate values
        dg = dc * i
        dg_raw = dg * tanh_derivative(gates[:, 2*self.hidden_size:3*self.hidden_size])
        
        # Concatenate gate gradients
        dgates = np.concatenate([di_raw, df_raw, dg_raw, do_raw], axis=1)
        
        # Gradients w.r.t. weights and biases
        self.dW_i = np.dot(x_dropped.T, dgates)
        self.dW_h = np.dot(h_dropped.T, dgates)
        self.db = np.sum(dgates, axis=0)
        
        # Gradients w.r.t. inputs
        dx_dropped = np.dot(dgates, self.W_i.T)
        dh_dropped = np.dot(dgates, self.W_h.T)
        
        # Apply dropout masks
        dx = dx_dropped * input_mask
        dh_prev = dh_dropped * recurrent_mask
        
        # Gradient w.r.t. previous cell state
        dc_prev = dc * f
        
        return dx, dh_prev, dc_prev

class LSTM:
    """Multi-layer LSTM implementation."""
    
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 dropout_rate=0.0, recurrent_dropout_rate=0.0, bidirectional=False):
        """
        Initialize LSTM.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for input connections
            recurrent_dropout_rate: Dropout rate for recurrent connections
            bidirectional: Whether to use bidirectional LSTM
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.bidirectional = bidirectional
        
        # Create LSTM cells
        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * (2 if bidirectional else 1)
            
            # Forward cells
            forward_cell = LSTMCell(
                layer_input_size, hidden_size, 
                dropout_rate, recurrent_dropout_rate
            )
            
            if bidirectional:
                # Backward cells
                backward_cell = LSTMCell(
                    layer_input_size, hidden_size,
                    dropout_rate, recurrent_dropout_rate
                )
                self.cells.append((forward_cell, backward_cell))
            else:
                self.cells.append(forward_cell)
    
    def forward(self, x, initial_state=None, training=True):
        """
        Forward pass through LSTM.
        
        Args:
            x: Input sequences (batch_size, seq_length, input_size)
            initial_state: Initial (h, c) states or None
            training: Whether in training mode
        
        Returns:
            outputs: Output sequences (batch_size, seq_length, hidden_size * num_directions)
            final_state: Final (h, c) states
        """
        batch_size, seq_length, _ = x.shape
        num_directions = 2 if self.bidirectional else 1
        
        # Initialize states
        if initial_state is None:
            h_states = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers * num_directions)]
            c_states = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers * num_directions)]
        else:
            h_states, c_states = initial_state
        
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
                    h_f, c_f = forward_cell.forward(
                        layer_input, h_states[layer*2], c_states[layer*2], training
                    )
                    h_states[layer*2] = h_f
                    c_states[layer*2] = c_f
                    
                    # Backward direction (process in reverse)
                    h_b, c_b = backward_cell.forward(
                        layer_input, h_states[layer*2+1], c_states[layer*2+1], training
                    )
                    h_states[layer*2+1] = h_b
                    c_states[layer*2+1] = c_b
                    
                    # Concatenate forward and backward outputs
                    layer_output = np.concatenate([h_f, h_b], axis=1)
                else:
                    cell = self.cells[layer]
                    h, c = cell.forward(layer_input, h_states[layer], c_states[layer], training)
                    h_states[layer] = h
                    c_states[layer] = c
                    layer_output = h
                
                layer_input = layer_output
            
            outputs.append(layer_output)
        
        # Stack outputs
        outputs = np.stack(outputs, axis=1)  # (batch_size, seq_length, hidden_size * num_directions)
        
        return outputs, (h_states, c_states)
    
    def get_parameters(self):
        """Get all trainable parameters."""
        params = []
        if self.bidirectional:
            for forward_cell, backward_cell in self.cells:
                params.extend([forward_cell.W_i, forward_cell.W_h, forward_cell.b])
                params.extend([backward_cell.W_i, backward_cell.W_h, backward_cell.b])
        else:
            for cell in self.cells:
                params.extend([cell.W_i, cell.W_h, cell.b])
        return params
    
    def get_gradients(self):
        """Get all gradients."""
        grads = []
        if self.bidirectional:
            for forward_cell, backward_cell in self.cells:
                grads.extend([forward_cell.dW_i, forward_cell.dW_h, forward_cell.db])
                grads.extend([backward_cell.dW_i, backward_cell.dW_h, backward_cell.db])
        else:
            for cell in self.cells:
                grads.extend([cell.dW_i, cell.dW_h, cell.db])
        return grads 