"""
Attention mechanisms for Seq2Seq models.
Includes Bahdanau (Additive) and Luong (Multiplicative) attention.
"""
import numpy as np
from ..core.activations import softmax, tanh
from ..core.utils import xavier_uniform

class BahdanauAttention:
    """
    Bahdanau (Additive) Attention mechanism.
    Reference: https://arxiv.org/abs/1409.0473
    """
    
    def __init__(self, encoder_hidden_size, decoder_hidden_size, attention_size):
        """
        Initialize Bahdanau attention.
        
        Args:
            encoder_hidden_size: Hidden size of encoder
            decoder_hidden_size: Hidden size of decoder
            attention_size: Size of attention hidden layer
        """
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_size = attention_size
        
        # Initialize weights
        self.W_a = xavier_uniform((encoder_hidden_size, attention_size))  # Encoder projection
        self.U_a = xavier_uniform((decoder_hidden_size, attention_size))  # Decoder projection
        self.v_a = xavier_uniform((attention_size, 1))  # Attention vector
        self.b_a = np.zeros((attention_size,))  # Bias
    
    def forward(self, encoder_outputs, decoder_hidden, encoder_mask=None):
        """
        Compute attention weights and context vector.
        
        Args:
            encoder_outputs: Encoder outputs (batch_size, seq_len, encoder_hidden_size)
            decoder_hidden: Current decoder hidden state (batch_size, decoder_hidden_size)
            encoder_mask: Mask for padded positions (batch_size, seq_len)
        
        Returns:
            context: Context vector (batch_size, encoder_hidden_size)
            attention_weights: Attention weights (batch_size, seq_len)
        """
        batch_size, seq_len, _ = encoder_outputs.shape
        
        # Project encoder outputs: (batch_size, seq_len, attention_size)
        encoder_proj = np.dot(encoder_outputs.reshape(-1, self.encoder_hidden_size), self.W_a)
        encoder_proj = encoder_proj.reshape(batch_size, seq_len, self.attention_size)
        
        # Project decoder hidden state: (batch_size, attention_size)
        decoder_proj = np.dot(decoder_hidden, self.U_a)
        
        # Add bias and compute tanh
        # Expand decoder_proj to match encoder_proj shape
        decoder_proj_expanded = np.expand_dims(decoder_proj, axis=1)  # (batch_size, 1, attention_size)
        
        # Compute attention scores
        scores = encoder_proj + decoder_proj_expanded + self.b_a  # (batch_size, seq_len, attention_size)
        scores = tanh(scores)
        
        # Apply attention vector
        scores = np.dot(scores.reshape(-1, self.attention_size), self.v_a)  # (batch_size * seq_len, 1)
        scores = scores.reshape(batch_size, seq_len)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if encoder_mask is not None:
            scores = scores + (1 - encoder_mask) * (-1e9)
        
        # Compute attention weights
        attention_weights = softmax(scores, axis=1)  # (batch_size, seq_len)
        
        # Compute context vector
        attention_weights_expanded = np.expand_dims(attention_weights, axis=2)  # (batch_size, seq_len, 1)
        context = np.sum(encoder_outputs * attention_weights_expanded, axis=1)  # (batch_size, encoder_hidden_size)
        
        return context, attention_weights
    
    def backward(self, dcontext, encoder_outputs, decoder_hidden, attention_weights, encoder_mask=None):
        """
        Backward pass through Bahdanau attention.
        
        Args:
            dcontext: Gradient w.r.t. context vector (batch_size, encoder_hidden_size)
            encoder_outputs: Encoder outputs (batch_size, seq_len, encoder_hidden_size)
            decoder_hidden: Current decoder hidden state (batch_size, decoder_hidden_size)
            attention_weights: Attention weights from forward pass (batch_size, seq_len)
            encoder_mask: Mask for padded positions (batch_size, seq_len)
        
        Returns:
            dencoder_outputs: Gradient w.r.t. encoder outputs
            ddecoder_hidden: Gradient w.r.t. decoder hidden state
        """
        batch_size, seq_len, _ = encoder_outputs.shape
        
        # Gradient w.r.t. attention weights
        dattention_weights = np.sum(dcontext[:, None, :] * encoder_outputs, axis=2)  # (batch_size, seq_len)
        
        # Gradient w.r.t. encoder outputs
        dencoder_outputs = dcontext[:, None, :] * attention_weights[:, :, None]  # (batch_size, seq_len, encoder_hidden_size)
        
        # Gradient w.r.t. attention scores (before softmax)
        dscores = dattention_weights * attention_weights - attention_weights * np.sum(dattention_weights * attention_weights, axis=1, keepdims=True)
        
        # Apply mask if provided
        if encoder_mask is not None:
            dscores = dscores * encoder_mask
        
        # Gradient w.r.t. attention vector
        encoder_proj = np.dot(encoder_outputs.reshape(-1, self.encoder_hidden_size), self.W_a)
        encoder_proj = encoder_proj.reshape(batch_size, seq_len, self.attention_size)
        decoder_proj = np.dot(decoder_hidden, self.U_a)
        decoder_proj_expanded = np.expand_dims(decoder_proj, axis=1)
        
        tanh_input = encoder_proj + decoder_proj_expanded + self.b_a
        dtanh_input = dscores[:, :, None] * (1 - tanh(tanh_input) ** 2)  # Gradient through tanh
        
        # Gradients w.r.t. parameters
        self.dv_a = np.sum(dtanh_input.reshape(-1, self.attention_size), axis=0, keepdims=True).T
        self.db_a = np.sum(dtanh_input, axis=(0, 1))
        
        # Gradients w.r.t. projections
        dencoder_proj = dtanh_input
        ddecoder_proj = np.sum(dtanh_input, axis=1)  # Sum over sequence length
        
        # Gradients w.r.t. weights
        self.dW_a = np.dot(encoder_outputs.reshape(-1, self.encoder_hidden_size).T, 
                          dencoder_proj.reshape(-1, self.attention_size))
        self.dU_a = np.dot(decoder_hidden.T, ddecoder_proj)
        
        # Gradients w.r.t. inputs
        dencoder_outputs += np.dot(dencoder_proj.reshape(-1, self.attention_size), self.W_a.T).reshape(batch_size, seq_len, self.encoder_hidden_size)
        ddecoder_hidden = np.dot(ddecoder_proj, self.U_a.T)
        
        return dencoder_outputs, ddecoder_hidden

class LuongAttention:
    """
    Luong (Multiplicative) Attention mechanism.
    Reference: https://arxiv.org/abs/1508.04025
    """
    
    def __init__(self, encoder_hidden_size, decoder_hidden_size, score_type='general'):
        """
        Initialize Luong attention.
        
        Args:
            encoder_hidden_size: Hidden size of encoder
            decoder_hidden_size: Hidden size of decoder
            score_type: Type of scoring function ('dot', 'general', 'concat')
        """
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.score_type = score_type
        
        # Initialize weights based on score type
        if score_type == 'general':
            self.W_a = xavier_uniform((decoder_hidden_size, encoder_hidden_size))
        elif score_type == 'concat':
            self.W_a = xavier_uniform((decoder_hidden_size + encoder_hidden_size, decoder_hidden_size))
            self.v_a = xavier_uniform((decoder_hidden_size, 1))
    
    def _score(self, encoder_outputs, decoder_hidden):
        """
        Compute attention scores.
        
        Args:
            encoder_outputs: Encoder outputs (batch_size, seq_len, encoder_hidden_size)
            decoder_hidden: Current decoder hidden state (batch_size, decoder_hidden_size)
        
        Returns:
            scores: Attention scores (batch_size, seq_len)
        """
        batch_size, seq_len, _ = encoder_outputs.shape
        
        if self.score_type == 'dot':
            # Dot product: decoder_hidden^T * encoder_outputs
            scores = np.sum(decoder_hidden[:, None, :] * encoder_outputs, axis=2)
        
        elif self.score_type == 'general':
            # General: decoder_hidden^T * W_a * encoder_outputs
            decoder_proj = np.dot(decoder_hidden, self.W_a)  # (batch_size, encoder_hidden_size)
            scores = np.sum(decoder_proj[:, None, :] * encoder_outputs, axis=2)
        
        elif self.score_type == 'concat':
            # Concat: v_a^T * tanh(W_a * [decoder_hidden; encoder_outputs])
            decoder_expanded = np.repeat(decoder_hidden[:, None, :], seq_len, axis=1)  # (batch_size, seq_len, decoder_hidden_size)
            concat = np.concatenate([decoder_expanded, encoder_outputs], axis=2)  # (batch_size, seq_len, decoder_hidden_size + encoder_hidden_size)
            
            hidden = tanh(np.dot(concat.reshape(-1, self.decoder_hidden_size + self.encoder_hidden_size), self.W_a))
            scores = np.dot(hidden, self.v_a).reshape(batch_size, seq_len)
        
        else:
            raise ValueError(f"Unknown score type: {self.score_type}")
        
        return scores
    
    def forward(self, encoder_outputs, decoder_hidden, encoder_mask=None):
        """
        Compute attention weights and context vector.
        
        Args:
            encoder_outputs: Encoder outputs (batch_size, seq_len, encoder_hidden_size)
            decoder_hidden: Current decoder hidden state (batch_size, decoder_hidden_size)
            encoder_mask: Mask for padded positions (batch_size, seq_len)
        
        Returns:
            context: Context vector (batch_size, encoder_hidden_size)
            attention_weights: Attention weights (batch_size, seq_len)
        """
        # Compute attention scores
        scores = self._score(encoder_outputs, decoder_hidden)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if encoder_mask is not None:
            scores = scores + (1 - encoder_mask) * (-1e9)
        
        # Compute attention weights
        attention_weights = softmax(scores, axis=1)  # (batch_size, seq_len)
        
        # Compute context vector
        attention_weights_expanded = np.expand_dims(attention_weights, axis=2)  # (batch_size, seq_len, 1)
        context = np.sum(encoder_outputs * attention_weights_expanded, axis=1)  # (batch_size, encoder_hidden_size)
        
        return context, attention_weights
    
    def backward(self, dcontext, encoder_outputs, decoder_hidden, attention_weights, encoder_mask=None):
        """
        Backward pass through Luong attention.
        
        Args:
            dcontext: Gradient w.r.t. context vector (batch_size, encoder_hidden_size)
            encoder_outputs: Encoder outputs (batch_size, seq_len, encoder_hidden_size)
            decoder_hidden: Current decoder hidden state (batch_size, decoder_hidden_size)
            attention_weights: Attention weights from forward pass (batch_size, seq_len)
            encoder_mask: Mask for padded positions (batch_size, seq_len)
        
        Returns:
            dencoder_outputs: Gradient w.r.t. encoder outputs
            ddecoder_hidden: Gradient w.r.t. decoder hidden state
        """
        batch_size, seq_len, _ = encoder_outputs.shape
        
        # Gradient w.r.t. attention weights
        dattention_weights = np.sum(dcontext[:, None, :] * encoder_outputs, axis=2)  # (batch_size, seq_len)
        
        # Gradient w.r.t. encoder outputs
        dencoder_outputs = dcontext[:, None, :] * attention_weights[:, :, None]  # (batch_size, seq_len, encoder_hidden_size)
        
        # Gradient w.r.t. attention scores (before softmax)
        dscores = dattention_weights * attention_weights - attention_weights * np.sum(dattention_weights * attention_weights, axis=1, keepdims=True)
        
        # Apply mask if provided
        if encoder_mask is not None:
            dscores = dscores * encoder_mask
        
        # Gradient w.r.t. decoder hidden state and parameters
        ddecoder_hidden = np.zeros_like(decoder_hidden)
        
        if self.score_type == 'dot':
            ddecoder_hidden = np.sum(dscores[:, :, None] * encoder_outputs, axis=1)
            dencoder_outputs += dscores[:, :, None] * decoder_hidden[:, None, :]
        
        elif self.score_type == 'general':
            decoder_proj = np.dot(decoder_hidden, self.W_a)
            ddecoder_proj = np.sum(dscores[:, :, None] * encoder_outputs, axis=1)
            ddecoder_hidden = np.dot(ddecoder_proj, self.W_a.T)
            self.dW_a = np.dot(decoder_hidden.T, ddecoder_proj)
            dencoder_outputs += dscores[:, :, None] * decoder_proj[:, None, :]
        
        elif self.score_type == 'concat':
            # More complex gradients for concat attention
            decoder_expanded = np.repeat(decoder_hidden[:, None, :], seq_len, axis=1)
            concat = np.concatenate([decoder_expanded, encoder_outputs], axis=2)
            
            hidden = tanh(np.dot(concat.reshape(-1, self.decoder_hidden_size + self.encoder_hidden_size), self.W_a))
            dhidden = dscores.reshape(-1, 1) * (1 - hidden ** 2)  # Gradient through tanh
            
            # Gradients w.r.t. parameters
            self.dW_a = np.dot(concat.reshape(-1, self.decoder_hidden_size + self.encoder_hidden_size).T, dhidden)
            self.dv_a = np.sum(dhidden, axis=0, keepdims=True).T
            
            # Gradients w.r.t. inputs
            dconcat = np.dot(dhidden, self.W_a.T).reshape(batch_size, seq_len, self.decoder_hidden_size + self.encoder_hidden_size)
            ddecoder_hidden = np.sum(dconcat[:, :, :self.decoder_hidden_size], axis=1)
            dencoder_outputs += dconcat[:, :, self.decoder_hidden_size:]
        
        return dencoder_outputs, ddecoder_hidden

def create_attention(attention_type, encoder_hidden_size, decoder_hidden_size, **kwargs):
    """
    Factory function to create attention mechanism.
    
    Args:
        attention_type: Type of attention ('bahdanau', 'luong')
        encoder_hidden_size: Hidden size of encoder
        decoder_hidden_size: Hidden size of decoder
        **kwargs: Additional arguments for specific attention types
    
    Returns:
        Attention mechanism instance
    """
    if attention_type.lower() == 'bahdanau':
        attention_size = kwargs.get('attention_size', min(encoder_hidden_size, decoder_hidden_size))
        return BahdanauAttention(encoder_hidden_size, decoder_hidden_size, attention_size)
    
    elif attention_type.lower() == 'luong':
        score_type = kwargs.get('score_type', 'general')
        return LuongAttention(encoder_hidden_size, decoder_hidden_size, score_type)
    
    else:
        raise ValueError(f"Unknown attention type: {attention_type}") 