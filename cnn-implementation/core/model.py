import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from .layers.base import Layer
from .losses import get_loss, Loss
from .optimizers import get_optimizer, Optimizer

class Sequential:
    """Sequential model for building and training neural networks."""
    
    def __init__(self, layers: Optional[List[Layer]] = None):
        """Initialize Sequential model.
        
        Args:
            layers: List of layers to add to the model
        """
        self.layers = layers or []
        self.loss_fn: Optional[Loss] = None
        self.optimizer: Optional[Optimizer] = None
        self.metrics_history: Dict[str, List[float]] = {'loss': []}
    
    def add(self, layer: Layer) -> None:
        """Add a layer to the model.
        
        Args:
            layer: Layer to add
        """
        self.layers.append(layer)
    
    def compile(self, optimizer: Union[str, Optimizer], loss: Union[str, Loss],
                metrics: Optional[List[str]] = None) -> None:
        """Configure the model for training.
        
        Args:
            optimizer: Name of optimizer or optimizer instance
            loss: Name of loss function or loss instance
            metrics: List of metric names to track
        """
        # Set optimizer
        if isinstance(optimizer, str):
            self.optimizer = get_optimizer(optimizer)
        else:
            self.optimizer = optimizer
        
        # Set loss function
        if isinstance(loss, str):
            self.loss_fn = get_loss(loss)
        else:
            self.loss_fn = loss
        
        # Initialize metrics history
        if metrics:
            for metric in metrics:
                self.metrics_history[metric] = []
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through the model.
        
        Args:
            x: Input data
            training: Whether in training mode
            
        Returns:
            Model output
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grad: np.ndarray) -> None:
        """Backward pass through the model.
        
        Args:
            grad: Gradient from loss function
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def _update_parameters(self) -> None:
        """Update model parameters using optimizer."""
        for layer in self.layers:
            params = layer.parameters
            if params:
                grads = layer.parameter_gradients
                updated_params = self.optimizer.update(params, grads)
                # Update layer parameters
                for key, value in updated_params.items():
                    setattr(layer, key, value)
    
    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: bool = True, patience: int = 5, min_delta: float = 1e-4) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            x: Training data
            y: Target data
            epochs: Number of epochs to train
            batch_size: Size of each training batch
            validation_data: Tuple of (validation_x, validation_y)
            verbose: Whether to print training progress
            patience: Number of epochs to wait for improvement before early stopping
            min_delta: Minimum change in validation loss to qualify as an improvement
            
        Returns:
            Training history
        """
        num_samples = len(x)
        num_batches = int(np.ceil(num_samples / batch_size))
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Training
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                
                # Get batch data
                batch_x = x[start_idx:end_idx]
                batch_y = y[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward(batch_x)
                
                # Compute loss
                batch_loss = self.loss_fn.forward(predictions, batch_y)
                epoch_loss += batch_loss
                
                # Backward pass
                grad = self.loss_fn.backward(predictions, batch_y)
                self.backward(grad)
                
                # Update parameters
                self._update_parameters()
            
            # Calculate epoch metrics
            epoch_loss /= num_batches
            self.metrics_history['loss'].append(epoch_loss)
            
            # Validation
            if validation_data is not None:
                val_x, val_y = validation_data
                val_predictions = self.forward(val_x, training=False)
                val_loss = self.loss_fn.forward(val_predictions, val_y)
                self.metrics_history.setdefault('val_loss', []).append(val_loss)
            
            # Print progress
            if verbose:
                status = f'Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}'
                if validation_data is not None:
                    status += f' - val_loss: {val_loss:.4f}'
                print(status)
            
            # Early stopping check
            if validation_data is not None:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f'Early stopping triggered after {epoch + 1} epochs')
                        break
        
        return self.metrics_history
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the model.
        
        Args:
            x: Test data
            y: True labels
            
        Returns:
            Test loss
        """
        predictions = self.forward(x, training=False)
        return self.loss_fn.forward(predictions, y)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            x: Input data
            
        Returns:
            Model predictions
        """
        return self.forward(x, training=False)