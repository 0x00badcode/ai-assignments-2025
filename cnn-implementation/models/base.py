"""Base model class for CNN architectures."""

import numpy as np
import pickle
import os
from typing import Optional, Dict, Any, Tuple
from core.model import Sequential
from utils.metrics import accuracy_score, confusion_matrix


class BaseModel(Sequential):
    """Enhanced base model with additional functionality."""
    
    def __init__(self, name: str = "BaseModel"):
        """Initialize BaseModel.
        
        Args:
            name: Model name
        """
        super().__init__()
        self.name = name
        self._is_compiled = False
        self.training_config = {}
    
    def summary(self) -> None:
        """Print model summary."""
        print(f"\nModel: {self.name}")
        print("=" * 70)
        print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15}")
        print("=" * 70)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_name = f"{layer.__class__.__name__}"
            
            # Get output shape (approximate)
            if hasattr(layer, 'filters'):
                output_shape = f"(None, {layer.filters}, ?, ?)"
            elif hasattr(layer, 'units'):
                output_shape = f"(None, {layer.units})"
            else:
                output_shape = "(None, ?)"
            
            # Count parameters
            layer_params = 0
            if hasattr(layer, 'weights') and layer.weights is not None:
                layer_params += np.prod(layer.weights.shape)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer_params += np.prod(layer.bias.shape)
            
            total_params += layer_params
            
            print(f"{layer_name:<25} {output_shape:<20} {layer_params:<15}")
        
        print("=" * 70)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {total_params:,}")
        print(f"Non-trainable params: 0")
    
    def save(self, filepath: str) -> None:
        """Save model to file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'name': self.name,
            'layers': [],
            'training_config': self.training_config,
            'metrics_history': self.metrics_history
        }
        
        # Save layer configurations and weights
        for layer in self.layers:
            layer_data = {
                'type': layer.__class__.__name__,
                'config': layer.__dict__.copy()
            }
            
            # Remove non-serializable items
            for key in ['input', 'output', 'activation', 'cache', 'gradients']:
                layer_data['config'].pop(key, None)
            
            model_data['layers'].append(layer_data)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """Load model from file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(name=model_data.get('name', 'LoadedModel'))
        model.training_config = model_data.get('training_config', {})
        model.metrics_history = model_data.get('metrics_history', {'loss': []})
        
        # Reconstruct layers (simplified - would need full layer reconstruction)
        print(f"Model loaded from {filepath}")
        print(f"Note: Layer weights need to be retrained or loaded separately")
        
        return model
    
    def evaluate_detailed(self, x: np.ndarray, y: np.ndarray, 
                         class_names: Optional[list] = None) -> Dict[str, Any]:
        """Detailed evaluation with multiple metrics.
        
        Args:
            x: Test data
            y: True labels
            class_names: Names of the classes
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(x)
        loss = self.evaluate(x, y)
        
        # Calculate accuracy
        acc = accuracy_score(y, predictions)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, predictions)
        
        results = {
            'loss': loss,
            'accuracy': acc,
            'confusion_matrix': cm,
            'num_samples': len(x),
            'num_classes': cm.shape[0] if len(cm.shape) == 2 else len(np.unique(y))
        }
        
        if class_names:
            results['class_names'] = class_names
        
        return results
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            x: Input data
            
        Returns:
            Predicted probabilities
        """
        return self.forward(x, training=False)
    
    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            x: Input data
            
        Returns:
            Predicted class labels
        """
        probabilities = self.predict_proba(x)
        return np.argmax(probabilities, axis=1)
    
    def get_layer(self, name: str):
        """Get layer by name.
        
        Args:
            name: Layer name
            
        Returns:
            Layer object or None if not found
        """
        for layer in self.layers:
            if hasattr(layer, 'name') and layer.name == name:
                return layer
        return None
    
    def freeze_layers(self, layer_names: list) -> None:
        """Freeze specified layers (prevent weight updates).
        
        Args:
            layer_names: List of layer names to freeze
        """
        for name in layer_names:
            layer = self.get_layer(name)
            if layer:
                layer.trainable = False
                print(f"Frozen layer: {name}")
    
    def unfreeze_layers(self, layer_names: list) -> None:
        """Unfreeze specified layers (allow weight updates).
        
        Args:
            layer_names: List of layer names to unfreeze
        """
        for name in layer_names:
            layer = self.get_layer(name)
            if layer:
                layer.trainable = True
                print(f"Unfrozen layer: {name}")
    
    def get_feature_extractor(self, layer_name: str) -> 'BaseModel':
        """Create a feature extractor up to a specified layer.
        
        Args:
            layer_name: Name of the layer to extract features from
            
        Returns:
            New model that outputs features from the specified layer
        """
        feature_model = BaseModel(name=f"{self.name}_features")
        
        for layer in self.layers:
            feature_model.add(layer)
            if hasattr(layer, 'name') and layer.name == layer_name:
                break
        
        return feature_model
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = 0
        trainable_params = 0
        
        for layer in self.layers:
            layer_params = 0
            if hasattr(layer, 'weights') and layer.weights is not None:
                layer_params += np.prod(layer.weights.shape)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer_params += np.prod(layer.bias.shape)
            
            total_params += layer_params
            
            if getattr(layer, 'trainable', True):
                trainable_params += layer_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        } 