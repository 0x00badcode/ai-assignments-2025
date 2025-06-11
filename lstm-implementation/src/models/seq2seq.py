"""
Sequence-to-Sequence model implementation for Neural Machine Translation.
Combines encoder, decoder, and attention mechanisms.
"""
import numpy as np
from .encoder import Encoder, BiLSTMEncoder
from .decoder import Decoder
from ..core.activations import softmax
from ..core.utils import create_mask

class Seq2SeqModel:
    """
    Complete Sequence-to-Sequence model for Neural Machine Translation.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=256, hidden_size=512,
                 num_layers=2, dropout_rate=0.1, recurrent_dropout_rate=0.1,
                 encoder_type='bilstm', attention_type='bahdanau', **attention_kwargs):
        """
        Initialize Seq2Seq model.
        
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            embedding_dim: Dimension of word embeddings
            hidden_size: Hidden size of encoder/decoder
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
            recurrent_dropout_rate: Recurrent dropout rate
            encoder_type: Type of encoder ('lstm', 'bilstm')
            attention_type: Type of attention ('bahdanau', 'luong', None)
            **attention_kwargs: Additional arguments for attention mechanism
        """
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.encoder_type = encoder_type
        self.attention_type = attention_type
        
        # Determine encoder hidden size
        encoder_hidden_size = hidden_size * (2 if encoder_type == 'bilstm' else 1)
        
        # Initialize encoder
        if encoder_type.lower() == 'bilstm':
            self.encoder = BiLSTMEncoder(
                vocab_size=src_vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                recurrent_dropout_rate=recurrent_dropout_rate
            )
        else:
            self.encoder = Encoder(
                vocab_size=src_vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                recurrent_dropout_rate=recurrent_dropout_rate,
                bidirectional=False
            )
        
        # Initialize decoder
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            encoder_hidden_size=encoder_hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            recurrent_dropout_rate=recurrent_dropout_rate,
            attention_type=attention_type,
            **attention_kwargs
        )
        
        # Special tokens
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2
        self.unk_token = 3
        
        # Training state
        self.training = True
    
    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        """
        Forward pass through the complete model.
        
        Args:
            src_ids: Source token IDs (batch_size, src_seq_length)
            tgt_ids: Target token IDs (batch_size, tgt_seq_length)
            src_mask: Source mask (batch_size, src_seq_length)
            tgt_mask: Target mask (batch_size, tgt_seq_length)
        
        Returns:
            logits: Output logits (batch_size, tgt_seq_length, tgt_vocab_size)
            attention_weights: Attention weights or None
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = create_mask(src_ids, self.pad_token)
        if tgt_mask is None:
            tgt_mask = create_mask(tgt_ids, self.pad_token)
        
        # Encoder forward pass
        encoder_outputs, encoder_final_state = self.encoder.forward(
            src_ids, src_mask, training=self.training
        )
        
        # Decoder forward pass (teacher forcing)
        decoder_logits, attention_weights = self.decoder.forward(
            tgt_ids, encoder_outputs, encoder_final_state, src_mask, training=self.training
        )
        
        return decoder_logits, attention_weights
    
    def translate(self, src_ids, src_mask=None, max_length=50, beam_size=1, temperature=1.0):
        """
        Translate source sequences to target sequences.
        
        Args:
            src_ids: Source token IDs (batch_size, src_seq_length)
            src_mask: Source mask (batch_size, src_seq_length)
            max_length: Maximum translation length
            beam_size: Beam search size (1 = greedy)
            temperature: Sampling temperature
        
        Returns:
            translations: Translated sequences (batch_size, translated_length)
            attention_weights: Attention weights or None
        """
        # Set to evaluation mode
        original_training = self.training
        self.training = False
        
        try:
            # Create mask if not provided
            if src_mask is None:
                src_mask = create_mask(src_ids, self.pad_token)
            
            # Encoder forward pass
            encoder_outputs, encoder_final_state = self.encoder.forward(
                src_ids, src_mask, training=False
            )
            
            if beam_size == 1:
                # Greedy decoding
                translations, attention_weights = self.decoder.generate(
                    encoder_outputs, encoder_final_state, src_mask,
                    max_length, self.start_token, self.end_token, temperature
                )
            else:
                # Beam search (simplified implementation)
                translations, attention_weights = self._beam_search(
                    encoder_outputs, encoder_final_state, src_mask,
                    max_length, beam_size
                )
            
            return translations, attention_weights
        
        finally:
            # Restore training mode
            self.training = original_training
    
    def _beam_search(self, encoder_outputs, encoder_final_state, encoder_mask, max_length, beam_size):
        """
        Simplified beam search implementation.
        
        Args:
            encoder_outputs: Encoder outputs
            encoder_final_state: Final encoder state
            encoder_mask: Encoder mask
            max_length: Maximum generation length
            beam_size: Beam size
        
        Returns:
            best_sequences: Best sequences (batch_size, seq_length)
            attention_weights: Attention weights or None
        """
        batch_size = encoder_outputs.shape[0]
        
        # For simplicity, we'll just do greedy search for now
        # Full beam search implementation would be quite complex
        return self.decoder.generate(
            encoder_outputs, encoder_final_state, encoder_mask,
            max_length, self.start_token, self.end_token, 1.0
        )
    
    def compute_loss(self, logits, targets, target_mask=None, label_smoothing=0.0):
        """
        Compute cross-entropy loss with optional label smoothing.
        
        Args:
            logits: Model output logits (batch_size, seq_length, vocab_size)
            targets: Target token IDs (batch_size, seq_length)
            target_mask: Target mask (batch_size, seq_length)
            label_smoothing: Label smoothing factor
        
        Returns:
            loss: Cross-entropy loss
            dlogits: Gradients w.r.t. logits
        """
        batch_size, seq_length, vocab_size = logits.shape
        
        # Create mask if not provided
        if target_mask is None:
            target_mask = create_mask(targets, self.pad_token)
        
        # Apply softmax
        probs = softmax(logits, axis=-1)
        
        # Create one-hot targets
        targets_one_hot = np.zeros((batch_size, seq_length, vocab_size))
        for i in range(batch_size):
            for j in range(seq_length):
                if target_mask[i, j] > 0:  # Only for non-padded positions
                    targets_one_hot[i, j, targets[i, j]] = 1.0
        
        # Apply label smoothing
        if label_smoothing > 0:
            targets_smooth = (1 - label_smoothing) * targets_one_hot + \
                           label_smoothing / vocab_size * np.ones_like(targets_one_hot)
            targets_one_hot = targets_smooth
        
        # Compute cross-entropy loss
        epsilon = 1e-8  # For numerical stability
        loss_per_token = -np.sum(targets_one_hot * np.log(probs + epsilon), axis=-1)
        
        # Apply mask and compute average loss
        masked_loss = loss_per_token * target_mask
        total_loss = np.sum(masked_loss)
        num_tokens = np.sum(target_mask)
        
        if num_tokens > 0:
            loss = total_loss / num_tokens
        else:
            loss = 0.0
        
        # Compute gradients
        dlogits = (probs - targets_one_hot) * target_mask[:, :, np.newaxis]
        if num_tokens > 0:
            dlogits = dlogits / num_tokens
        
        return loss, dlogits
    
    def train_step(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None, label_smoothing=0.0):
        """
        Single training step.
        
        Args:
            src_ids: Source token IDs (batch_size, src_seq_length)
            tgt_ids: Target token IDs (batch_size, tgt_seq_length)
            src_mask: Source mask (batch_size, src_seq_length)
            tgt_mask: Target mask (batch_size, tgt_seq_length)
            label_smoothing: Label smoothing factor
        
        Returns:
            loss: Training loss
            gradients: Model gradients
        """
        # Set to training mode
        self.training = True
        
        # Forward pass
        logits, attention_weights = self.forward(src_ids, tgt_ids, src_mask, tgt_mask)
        
        # Compute loss
        loss, dlogits = self.compute_loss(logits, tgt_ids, tgt_mask, label_smoothing)
        
        # Backward pass
        self.backward(dlogits)
        
        # Get gradients
        gradients = self.get_gradients()
        
        return loss, gradients
    
    def backward(self, dlogits):
        """
        Backward pass through the model.
        
        Args:
            dlogits: Gradients w.r.t. output logits
        """
        # Backward through decoder
        dencoder_outputs = self.decoder.backward(dlogits)
        
        # Backward through encoder
        self.encoder.backward(dencoder_outputs)
    
    def get_parameters(self):
        """Get all model parameters."""
        params = []
        params.extend(self.encoder.get_parameters())
        params.extend(self.decoder.get_parameters())
        return params
    
    def get_gradients(self):
        """Get all model gradients."""
        grads = []
        grads.extend(self.encoder.get_gradients())
        grads.extend(self.decoder.get_gradients())
        return grads
    
    def set_parameters(self, parameters):
        """Set model parameters."""
        encoder_params = self.encoder.get_parameters()
        decoder_params = self.decoder.get_parameters()
        
        num_encoder_params = len(encoder_params)
        
        # Set encoder parameters
        for i, param in enumerate(parameters[:num_encoder_params]):
            encoder_params[i][:] = param
        
        # Set decoder parameters
        decoder_start_idx = num_encoder_params
        for i, param in enumerate(parameters[decoder_start_idx:]):
            decoder_params[i][:] = param
    
    def save_model(self, filepath):
        """Save model parameters to file."""
        parameters = self.get_parameters()
        np.savez(filepath, *parameters)
    
    def load_model(self, filepath):
        """Load model parameters from file."""
        data = np.load(filepath)
        parameters = [data[f'arr_{i}'] for i in range(len(data.files))]
        self.set_parameters(parameters)
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.training = False
    
    def train_mode(self):
        """Set model to training mode."""
        self.training = True

def create_seq2seq_model(config):
    """
    Factory function to create Seq2Seq model from configuration.
    
    Args:
        config: Dictionary containing model configuration
    
    Returns:
        Seq2SeqModel instance
    """
    return Seq2SeqModel(**config) 