"""
Multi-Layer Perceptron (MLP) implementation
Main model class with flexible architecture and comprehensive training capabilities
"""

import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from typing import List, Union, Dict, Any, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor
import threading

from .layers import Dense, Dropout, BatchNormalization
from .activations import get_activation
from .losses import get_loss
from ..optimizers import SGD, Momentum, RMSProp, Adam
from ..regularization import L1Regularization, L2Regularization, ElasticNetRegularization
from ..evaluation import ConfusionMatrix


class StoppingCriteria:
    """Early stopping criteria for training"""
    
    def __init__(self, patience=10, min_delta=1e-4, monitor='loss', mode='min'):
        """
        Initialize stopping criteria
        
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change to qualify as an improvement
            monitor: Metric to monitor ('loss', 'accuracy', 'val_loss', 'val_accuracy')
            mode: 'min' or 'max' depending on whether monitored metric should be minimized or maximized
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
    
    def should_stop(self, current_value, epoch):
        """Check if training should stop"""
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True
        
        return False


class MLP:
    """Multi-Layer Perceptron implementation"""
    
    def __init__(self, task_type='classification'):
        """
        Initialize MLP
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.layers = []
        self.compiled = False
        self.trained = False
        
        # Training components
        self.optimizer = None
        self.loss_function = None
        self.regularization = None
        
        # Training history
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Model metadata
        self.input_shape = None
        self.output_shape = None
        self.training_time = 0
        
    def add_layer(self, layer):
        """
        Add a layer to the model
        
        Args:
            layer: Layer instance (Dense, Dropout, BatchNormalization)
        """
        if self.compiled:
            raise ValueError("Cannot add layers after model is compiled")
        
        self.layers.append(layer)
        return self
    
    def add_dense(self, units, activation='relu', weight_init='xavier', 
                  bias_init='zeros', use_bias=True):
        """
        Add a dense layer to the model
        
        Args:
            units: Number of neurons
            activation: Activation function
            weight_init: Weight initialization method
            bias_init: Bias initialization method
            use_bias: Whether to use bias
        """
        if len(self.layers) == 0:
            raise ValueError("First layer must specify input_size. Use add_dense with input_size parameter.")
        
        # Get input size from previous layer
        prev_layer = self.layers[-1]
        if isinstance(prev_layer, Dense):
            input_size = prev_layer.output_size
        else:
            # For other layer types, we need to track the size
            input_size = getattr(prev_layer, 'output_size', None)
            if input_size is None:
                raise ValueError("Cannot determine input size for this layer")
        
        layer = Dense(input_size, units, activation, weight_init, bias_init, use_bias)
        return self.add_layer(layer)
    
    def add_first_dense(self, input_size, units, activation='relu', 
                       weight_init='xavier', bias_init='zeros', use_bias=True):
        """
        Add the first dense layer with specified input size
        
        Args:
            input_size: Number of input features
            units: Number of neurons
            activation: Activation function
            weight_init: Weight initialization method
            bias_init: Bias initialization method
            use_bias: Whether to use bias
        """
        if len(self.layers) > 0:
            raise ValueError("add_first_dense should only be used for the first layer")
        
        layer = Dense(input_size, units, activation, weight_init, bias_init, use_bias)
        return self.add_layer(layer)
    
    def add_dropout(self, rate=0.5):
        """Add a dropout layer"""
        layer = Dropout(rate)
        layer.output_size = getattr(self.layers[-1], 'output_size', None)
        return self.add_layer(layer)
    
    def add_batch_norm(self):
        """Add batch normalization layer"""
        if len(self.layers) == 0:
            raise ValueError("Cannot add batch normalization as first layer")
        
        prev_layer = self.layers[-1]
        if isinstance(prev_layer, Dense):
            num_features = prev_layer.output_size
        else:
            num_features = getattr(prev_layer, 'output_size', None)
            if num_features is None:
                raise ValueError("Cannot determine number of features for batch normalization")
        
        layer = BatchNormalization(num_features)
        layer.output_size = num_features
        return self.add_layer(layer)
    
    def compile(self, optimizer='adam', loss=None, regularization=None, 
                learning_rate=0.001, **optimizer_kwargs):
        """
        Compile the model
        
        Args:
            optimizer: Optimizer name or instance
            loss: Loss function name or instance
            regularization: Regularization instance
            learning_rate: Learning rate for optimizer
            **optimizer_kwargs: Additional optimizer parameters
        """
        if len(self.layers) == 0:
            raise ValueError("Model must have at least one layer")
        
        # Set default loss based on task type
        if loss is None:
            if self.task_type == 'classification':
                # Determine if binary or multi-class
                output_layer = self.layers[-1]
                if isinstance(output_layer, Dense):
                    if output_layer.output_size == 1:
                        loss = 'binary_crossentropy'
                    else:
                        loss = 'categorical_crossentropy'
                else:
                    loss = 'categorical_crossentropy'
            else:  # regression
                loss = 'mse'
        
        # Set up optimizer
        if isinstance(optimizer, str):
            optimizer_map = {
                'sgd': SGD,
                'momentum': Momentum,
                'rmsprop': RMSProp,
                'adam': Adam
            }
            if optimizer.lower() in optimizer_map:
                self.optimizer = optimizer_map[optimizer.lower()](learning_rate, **optimizer_kwargs)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            self.optimizer = optimizer
        
        # Set up loss function
        if isinstance(loss, str):
            self.loss_function = get_loss(loss)
        else:
            self.loss_function = loss
        
        # Set regularization
        self.regularization = regularization
        
        # Set input/output shapes
        first_layer = self.layers[0]
        last_layer = self.layers[-1]
        
        if isinstance(first_layer, Dense):
            self.input_shape = first_layer.input_size
        
        if isinstance(last_layer, Dense):
            self.output_shape = last_layer.output_size
        
        self.compiled = True
        return self
    
    def forward(self, X):
        """
        Forward pass through the network
        
        Args:
            X: Input data
            
        Returns:
            Output predictions
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grad_output):
        """
        Backward pass through the network
        
        Args:
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            Gradient w.r.t. input
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def _compute_loss(self, y_true, y_pred):
        """Compute total loss including regularization"""
        # Base loss
        loss = self.loss_function.forward(y_true, y_pred)
        
        # Add regularization
        if self.regularization is not None:
            reg_penalty = 0
            for layer in self.layers:
                if isinstance(layer, Dense):
                    reg_penalty += self.regularization.compute_penalty(layer.weights)
            loss += reg_penalty
        
        return loss
    
    def _update_parameters(self):
        """Update parameters using optimizer"""
        for layer in self.layers:
            if hasattr(layer, 'get_gradients'):
                gradients = layer.get_gradients()
                
                # Add regularization to weight gradients
                if self.regularization is not None and isinstance(layer, Dense):
                    if 'weights' in gradients:
                        gradients['weights'] += self.regularization.compute_gradient(layer.weights)
                
                self.optimizer.update(layer, gradients)
    
    def _set_training_mode(self, training=True):
        """Set training mode for all layers"""
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(training)
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        if not self.compiled:
            raise ValueError("Model must be compiled before prediction")
        
        # Set to evaluation mode
        self._set_training_mode(False)
        
        # Forward pass
        predictions = self.forward(X)
        
        return predictions
    
    def predict_classes(self, X):
        """
        Predict classes for classification tasks
        
        Args:
            X: Input data
            
        Returns:
            Predicted class labels
        """
        if self.task_type != 'classification':
            raise ValueError("predict_classes is only available for classification tasks")
        
        predictions = self.predict(X)
        
        if predictions.shape[1] == 1:  # Binary classification
            return (predictions > 0.5).astype(int).flatten()
        else:  # Multi-class classification
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for classification tasks
        
        Args:
            X: Input data
            
        Returns:
            Predicted probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        return self.predict(X)
    
    def evaluate(self, X, y, batch_size=None):
        """
        Evaluate the model
        
        Args:
            X: Input data
            y: True labels/values
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.compiled:
            raise ValueError("Model must be compiled before evaluation")
        
        predictions = self.predict(X)
        loss = self._compute_loss(y, predictions)
        
        metrics = {'loss': loss}
        
        if self.task_type == 'classification':
            if predictions.shape[1] == 1:  # Binary classification
                pred_classes = (predictions > 0.5).astype(int).flatten()
                y_true = y.flatten()
            else:  # Multi-class classification
                pred_classes = np.argmax(predictions, axis=1)
                if y.ndim == 2:  # One-hot encoded
                    y_true = np.argmax(y, axis=1)
                else:
                    y_true = y
            
            accuracy = np.mean(pred_classes == y_true)
            metrics['accuracy'] = accuracy
        
        else:  # Regression
            mse = np.mean((y - predictions) ** 2)
            mae = np.mean(np.abs(y - predictions))
            metrics.update({'mse': mse, 'mae': mae})
        
        return metrics
    
    def fit(self, X, y, epochs=100, batch_size=32, validation_data=None, 
            verbose=1, early_stopping=None, shuffle=True):
        """
        Train the model
        
        Args:
            X: Training input data
            y: Training target data
            epochs: Number of training epochs
            batch_size: Batch size
            validation_data: Tuple (X_val, y_val) for validation
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
            early_stopping: StoppingCriteria instance
            shuffle: Whether to shuffle training data
            
        Returns:
            Training history
        """
        if not self.compiled:
            raise ValueError("Model must be compiled before training")
        
        start_time = time.time()
        X = np.array(X)
        y = np.array(y)
        
        # Initialize history
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            # Set training mode
            self._set_training_mode(True)
            
            # Shuffle data
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
            
            epoch_losses = []
            epoch_accuracies = []
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward(X_batch)
                
                # Compute loss
                loss = self._compute_loss(y_batch, predictions)
                epoch_losses.append(loss)
                
                # Compute accuracy for classification
                if self.task_type == 'classification':
                    if predictions.shape[1] == 1:  # Binary
                        pred_classes = (predictions > 0.5).astype(int).flatten()
                        y_true = y_batch.flatten()
                    else:  # Multi-class
                        pred_classes = np.argmax(predictions, axis=1)
                        if y_batch.ndim == 2:
                            y_true = np.argmax(y_batch, axis=1)
                        else:
                            y_true = y_batch
                    
                    accuracy = np.mean(pred_classes == y_true)
                    epoch_accuracies.append(accuracy)
                
                # Backward pass
                grad_output = self.loss_function.backward(y_batch, predictions)
                self.backward(grad_output)
                
                # Update parameters
                self._update_parameters()
            
            # Record training metrics
            avg_loss = np.mean(epoch_losses)
            self.history['loss'].append(avg_loss)
            
            if self.task_type == 'classification':
                avg_accuracy = np.mean(epoch_accuracies)
                self.history['accuracy'].append(avg_accuracy)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_metrics = self.evaluate(X_val, y_val)
                self.history['val_loss'].append(val_metrics['loss'])
                if 'accuracy' in val_metrics:
                    self.history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Verbose output
            if verbose > 0:
                if verbose == 1:
                    # Progress bar style
                    progress = (epoch + 1) / epochs
                    bar_length = 30
                    filled_length = int(bar_length * progress)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    
                    metrics_str = f"loss: {avg_loss:.4f}"
                    if self.task_type == 'classification':
                        metrics_str += f", acc: {avg_accuracy:.4f}"
                    
                    if validation_data is not None:
                        metrics_str += f", val_loss: {val_metrics['loss']:.4f}"
                        if 'accuracy' in val_metrics:
                            metrics_str += f", val_acc: {val_metrics['accuracy']:.4f}"
                    
                    print(f"\rEpoch {epoch+1}/{epochs} [{bar}] {metrics_str}", end='')
                    
                elif verbose == 2:
                    # One line per epoch
                    metrics_str = f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}"
                    if self.task_type == 'classification':
                        metrics_str += f" - acc: {avg_accuracy:.4f}"
                    
                    if validation_data is not None:
                        metrics_str += f" - val_loss: {val_metrics['loss']:.4f}"
                        if 'accuracy' in val_metrics:
                            metrics_str += f" - val_acc: {val_metrics['accuracy']:.4f}"
                    
                    print(metrics_str)
            
            # Early stopping
            if early_stopping is not None:
                monitor_value = None
                if early_stopping.monitor == 'loss':
                    monitor_value = avg_loss
                elif early_stopping.monitor == 'accuracy' and self.task_type == 'classification':
                    monitor_value = avg_accuracy
                elif early_stopping.monitor == 'val_loss' and validation_data is not None:
                    monitor_value = val_metrics['loss']
                elif early_stopping.monitor == 'val_accuracy' and validation_data is not None and 'accuracy' in val_metrics:
                    monitor_value = val_metrics['accuracy']
                
                if monitor_value is not None and early_stopping.should_stop(monitor_value, epoch):
                    if verbose > 0:
                        print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        if verbose > 0:
            print()  # New line after training
        
        self.training_time = time.time() - start_time
        self.trained = True
        
        return self.history
    
    def plot_history(self, save_path=None, figsize=(12, 4)):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not self.history['loss']:
            print("No training history to plot")
            return
        
        n_plots = 1 if self.task_type == 'regression' else 2
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        if n_plots == 1:
            axes = [axes]
        
        # Loss plot
        axes[0].plot(self.history['loss'], label='Training Loss')
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot (classification only)
        if self.task_type == 'classification' and n_plots > 1:
            axes[1].plot(self.history['accuracy'], label='Training Accuracy')
            if self.history['val_accuracy']:
                axes[1].plot(self.history['val_accuracy'], label='Validation Accuracy')
            axes[1].set_title('Model Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save(self, filepath):
        """
        Save the model to file
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'task_type': self.task_type,
            'compiled': self.compiled,
            'trained': self.trained,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'training_time': self.training_time,
            'history': self.history,
            'layers': [],
            'optimizer_config': self.optimizer.get_config() if self.optimizer else None,
            'loss_function': self.loss_function.__class__.__name__ if self.loss_function else None,
            'regularization': self.regularization.__class__.__name__ if self.regularization else None
        }
        
        # Save layer configurations and parameters
        for layer in self.layers:
            layer_data = {
                'type': layer.__class__.__name__,
                'params': layer.get_params()
            }
            
            # Save layer-specific configurations
            if isinstance(layer, Dense):
                layer_data['config'] = {
                    'input_size': layer.input_size,
                    'output_size': layer.output_size,
                    'activation': layer.activation.__class__.__name__,
                    'use_bias': layer.use_bias
                }
            elif isinstance(layer, Dropout):
                layer_data['config'] = {
                    'rate': layer.rate
                }
            elif isinstance(layer, BatchNormalization):
                layer_data['config'] = {
                    'num_features': layer.num_features,
                    'momentum': layer.momentum,
                    'epsilon': layer.epsilon
                }
            
            model_data['layers'].append(layer_data)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from file
        
        Args:
            filepath: Path to the model file
            
        Returns:
            Loaded MLP model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new model instance
        model = cls(task_type=model_data['task_type'])
        
        # Restore layers
        for layer_data in model_data['layers']:
            layer_type = layer_data['type']
            layer_config = layer_data['config']
            layer_params = layer_data['params']
            
            if layer_type == 'Dense':
                layer = Dense(
                    layer_config['input_size'],
                    layer_config['output_size'],
                    layer_config['activation'],
                    use_bias=layer_config['use_bias']
                )
            elif layer_type == 'Dropout':
                layer = Dropout(layer_config['rate'])
                layer.output_size = layer_config.get('output_size')
            elif layer_type == 'BatchNormalization':
                layer = BatchNormalization(
                    layer_config['num_features'],
                    layer_config['momentum'],
                    layer_config['epsilon']
                )
                layer.output_size = layer_config['num_features']
            
            layer.set_params(layer_params)
            model.add_layer(layer)
        
        # Restore other attributes
        model.compiled = model_data['compiled']
        model.trained = model_data['trained']
        model.input_shape = model_data['input_shape']
        model.output_shape = model_data['output_shape']
        model.training_time = model_data['training_time']
        model.history = model_data['history']
        
        print(f"Model loaded from {filepath}")
        return model
    
    def summary(self):
        """Print model summary"""
        print("Model Summary")
        print("=" * 50)
        print(f"Task Type: {self.task_type}")
        print(f"Input Shape: {self.input_shape}")
        print(f"Output Shape: {self.output_shape}")
        print(f"Compiled: {self.compiled}")
        print(f"Trained: {self.trained}")
        
        if self.trained:
            print(f"Training Time: {self.training_time:.2f} seconds")
        
        print("\nLayer Configuration:")
        print("-" * 50)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_params = 0
            if isinstance(layer, Dense):
                layer_params = layer.weights.size
                if layer.use_bias:
                    layer_params += layer.bias.size
            elif isinstance(layer, BatchNormalization):
                layer_params = layer.gamma.size + layer.beta.size
            
            total_params += layer_params
            print(f"Layer {i+1}: {layer} - Parameters: {layer_params}")
        
        print("-" * 50)
        print(f"Total Parameters: {total_params}")
        
        if self.optimizer:
            print(f"Optimizer: {self.optimizer}")
        
        if self.loss_function:
            print(f"Loss Function: {self.loss_function.__class__.__name__}")
        
        if self.regularization:
            print(f"Regularization: {self.regularization}")
    
    def get_weights(self):
        """Get all model weights"""
        weights = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                weights.append({
                    'weights': layer.weights.copy(),
                    'bias': layer.bias.copy() if layer.use_bias else None
                })
        return weights
    
    def set_weights(self, weights):
        """Set model weights"""
        dense_idx = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                if dense_idx < len(weights):
                    layer.weights = weights[dense_idx]['weights'].copy()
                    if layer.use_bias and weights[dense_idx]['bias'] is not None:
                        layer.bias = weights[dense_idx]['bias'].copy()
                    dense_idx += 1
    
    def __repr__(self):
        return f"MLP(task_type='{self.task_type}', layers={len(self.layers)}, compiled={self.compiled})" 