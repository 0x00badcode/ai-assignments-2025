#!/usr/bin/env python3
"""
CNN Face Recognition Training
============================

Main training script for CNN face recognition implementation.
This script provides stable CNN training with proper gradient computation
using the LFW dataset from the archive directory.

Features:
- Stable CNN implementation with gradient clipping
- Multiple optimizers (Adam, SGD)
- LFW dataset from archive/lfw-deepfunneled/
- Comprehensive evaluation and reporting
- Proper numerical stability

Author: CNN Implementation Team
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from PIL import Image
from datetime import datetime

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.metrics import accuracy_score, confusion_matrix


class CNNLayer:
    """CNN layer with stable backward pass."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Better weight initialization (Xavier/He initialization)
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        self.weights = np.random.uniform(-limit, limit, 
                                       (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(out_channels)
        
        # For gradients
        self.weight_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)
        
    def forward(self, x):
        """Forward pass."""
        self.input = x
        batch_size, in_channels, in_height, in_width = x.shape
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Pad input if needed
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), 
                                (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x
        
        # Initialize output
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Convolution operation
        for n in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Extract patch and compute convolution
                        patch = x_padded[n, :, h_start:h_end, w_start:w_end]
                        output[n, c_out, h, w] = np.sum(patch * self.weights[c_out]) + self.bias[c_out]
        
        self.output = output
        return output
    
    def backward(self, grad_output):
        """Backward pass with gradient clipping."""
        batch_size, in_channels, in_height, in_width = self.input.shape
        
        # Pad input if needed
        if self.padding > 0:
            x_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), 
                                         (self.padding, self.padding)), mode='constant')
        else:
            x_padded = self.input
        
        # Initialize gradients
        self.weight_grad.fill(0)
        self.bias_grad.fill(0)
        grad_input = np.zeros_like(x_padded)
        
        out_height, out_width = grad_output.shape[2], grad_output.shape[3]
        
        # Compute gradients
        for n in range(batch_size):
            for c_out in range(self.out_channels):
                # Bias gradient
                self.bias_grad[c_out] += np.sum(grad_output[n, c_out])
                
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        grad_val = grad_output[n, c_out, h, w]
                        
                        # Weight gradient
                        patch = x_padded[n, :, h_start:h_end, w_start:w_end]
                        self.weight_grad[c_out] += grad_val * patch
                        
                        # Input gradient
                        grad_input[n, :, h_start:h_end, w_start:w_end] += \
                            grad_val * self.weights[c_out]
        
        # Gradient clipping to prevent overflow
        max_grad_norm = 1.0
        weight_grad_norm = np.sqrt(np.sum(self.weight_grad ** 2))
        if weight_grad_norm > max_grad_norm:
            self.weight_grad = self.weight_grad * max_grad_norm / weight_grad_norm
        
        bias_grad_norm = np.sqrt(np.sum(self.bias_grad ** 2))
        if bias_grad_norm > max_grad_norm:
            self.bias_grad = self.bias_grad * max_grad_norm / bias_grad_norm
        
        # Remove padding from input gradient if needed
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        return grad_input


class DenseLayer:
    """Dense layer with stable backward pass."""
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # Better weight initialization
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros(output_size)
        
        # For gradients
        self.weight_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)
    
    def forward(self, x):
        """Forward pass."""
        # Flatten if needed
        if len(x.shape) > 2:
            self.input_shape = x.shape
            x = x.reshape(x.shape[0], -1)
        else:
            self.input_shape = x.shape
        
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.output
    
    def backward(self, grad_output):
        """Backward pass with gradient clipping."""
        # Compute gradients
        self.weight_grad = np.dot(self.input.T, grad_output)
        self.bias_grad = np.sum(grad_output, axis=0)
        
        # Gradient clipping
        max_grad_norm = 1.0
        weight_grad_norm = np.sqrt(np.sum(self.weight_grad ** 2))
        if weight_grad_norm > max_grad_norm:
            self.weight_grad = self.weight_grad * max_grad_norm / weight_grad_norm
        
        bias_grad_norm = np.sqrt(np.sum(self.bias_grad ** 2))
        if bias_grad_norm > max_grad_norm:
            self.bias_grad = self.bias_grad * max_grad_norm / bias_grad_norm
        
        # Input gradient
        grad_input = np.dot(grad_output, self.weights.T)
        
        # Reshape back if needed
        if len(self.input_shape) > 2:
            grad_input = grad_input.reshape(self.input_shape)
        
        return grad_input


class Activation:
    """Activation functions with numerical stability."""
    
    @staticmethod
    def relu_forward(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_backward(x, grad_output):
        return grad_output * (x > 0)
    
    @staticmethod
    def softmax_forward(x):
        # Improved numerical stability
        x_max = np.max(x, axis=1, keepdims=True)
        x_shifted = x - x_max
        # Clip to prevent overflow
        x_shifted = np.clip(x_shifted, -500, 500)
        exp_x = np.exp(x_shifted)
        softmax_out = exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-15)
        return softmax_out
    
    @staticmethod
    def softmax_backward(softmax_output, grad_output):
        # Simplified gradient for cross-entropy + softmax
        return grad_output


class CNN:
    """CNN model with stable backward pass."""
    
    def __init__(self, num_classes):
        self.layers = []
        self.activations = []
        self.num_classes = num_classes
        
        # Build simple but stable CNN
        self.conv1 = CNNLayer(3, 16, kernel_size=5, stride=2, padding=2)  # 32x32 -> 16x16
        self.conv2 = CNNLayer(16, 32, kernel_size=5, stride=2, padding=2)  # 16x16 -> 8x8
        self.conv3 = CNNLayer(32, 64, kernel_size=4, stride=2, padding=1)  # 8x8 -> 4x4
        
        # Dense layers
        self.dense1 = DenseLayer(64 * 4 * 4, 128)
        self.dense2 = DenseLayer(128, num_classes)
        
        self.training = True
    
    def forward(self, x):
        """Forward pass."""
        # Convolutional layers with ReLU
        x = self.conv1.forward(x)
        x = Activation.relu_forward(x)
        self.conv1_output = x
        
        x = self.conv2.forward(x)
        x = Activation.relu_forward(x)
        self.conv2_output = x
        
        x = self.conv3.forward(x)
        x = Activation.relu_forward(x)
        self.conv3_output = x
        
        # Dense layers
        x = self.dense1.forward(x)
        x = Activation.relu_forward(x)
        self.dense1_output = x
        
        # Light dropout simulation during training
        if self.training:
            dropout_mask = np.random.binomial(1, 0.8, x.shape)
            x = x * dropout_mask / 0.8
            self.dropout_mask = dropout_mask
        
        x = self.dense2.forward(x)
        x = Activation.softmax_forward(x)
        
        return x
    
    def backward(self, grad_output):
        """Backward pass."""
        # Softmax + dense2
        grad = Activation.softmax_backward(self.dense2.output, grad_output)
        grad = self.dense2.backward(grad)
        
        # Dropout backward
        if self.training:
            grad = grad * self.dropout_mask / 0.8
        
        # Dense1 + ReLU
        grad = Activation.relu_backward(self.dense1_output, grad)
        grad = self.dense1.backward(grad)
        
        # Conv3 + ReLU
        grad = Activation.relu_backward(self.conv3_output, grad)
        grad = self.conv3.backward(grad)
        
        # Conv2 + ReLU
        grad = Activation.relu_backward(self.conv2_output, grad)
        grad = self.conv2.backward(grad)
        
        # Conv1 + ReLU
        grad = Activation.relu_backward(self.conv1_output, grad)
        grad = self.conv1.backward(grad)
    
    def get_parameters(self):
        """Get all parameters."""
        params = []
        params.extend([self.conv1.weights, self.conv1.bias])
        params.extend([self.conv2.weights, self.conv2.bias])
        params.extend([self.conv3.weights, self.conv3.bias])
        params.extend([self.dense1.weights, self.dense1.bias])
        params.extend([self.dense2.weights, self.dense2.bias])
        return params
    
    def get_gradients(self):
        """Get all gradients."""
        grads = []
        grads.extend([self.conv1.weight_grad, self.conv1.bias_grad])
        grads.extend([self.conv2.weight_grad, self.conv2.bias_grad])
        grads.extend([self.conv3.weight_grad, self.conv3.bias_grad])
        grads.extend([self.dense1.weight_grad, self.dense1.bias_grad])
        grads.extend([self.dense2.weight_grad, self.dense2.bias_grad])
        return grads


class Optimizer:
    """Optimizer implementations with stability improvements."""
    
    def __init__(self, learning_rate=0.001, optimizer_type='adam'):
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        
        if optimizer_type == 'adam':
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.m = []
            self.v = []
            self.t = 0
    
    def initialize(self, params):
        """Initialize optimizer state."""
        if self.optimizer_type == 'adam':
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
    
    def update(self, params, grads):
        """Update parameters with gradient clipping."""
        if not hasattr(self, 'm') or len(self.m) == 0:
            self.initialize(params)
        
        # Global gradient clipping
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        clip_norm = 5.0
        if total_norm > clip_norm:
            grads = [g * clip_norm / total_norm for g in grads]
        
        if self.optimizer_type == 'sgd':
            for param, grad in zip(params, grads):
                param -= self.learning_rate * grad
        
        elif self.optimizer_type == 'adam':
            self.t += 1
            for i, (param, grad) in enumerate(zip(params, grads)):
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class CrossEntropyLoss:
    """Cross-entropy loss with numerical stability."""
    
    def forward(self, predictions, targets):
        """Forward pass."""
        batch_size = predictions.shape[0]
        
        # Convert targets to one-hot if needed
        if len(targets.shape) == 1:
            one_hot = np.zeros((batch_size, predictions.shape[1]))
            one_hot[np.arange(batch_size), targets] = 1
            targets = one_hot
        
        # Compute loss with better numerical stability
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        loss = -np.mean(np.sum(targets * np.log(predictions), axis=1))
        
        # Check for NaN/inf and handle
        if np.isnan(loss) or np.isinf(loss):
            loss = 10.0  # Return a reasonable default loss
        
        return loss
    
    def backward(self, predictions, targets):
        """Backward pass."""
        batch_size = predictions.shape[0]
        
        # Convert targets to one-hot if needed
        if len(targets.shape) == 1:
            one_hot = np.zeros((batch_size, predictions.shape[1]))
            one_hot[np.arange(batch_size), targets] = 1
            targets = one_hot
        
        # Simplified gradient for softmax + cross-entropy
        grad = (predictions - targets) / batch_size
        
        return grad


class CNNTrainer:
    """Main CNN training class with stable implementation."""
    
    def __init__(self):
        self.start_time = time.time()
        self.results_dir = "training_results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.all_results = {}
    
    def log_message(self, message):
        """Log message with timestamp."""
        elapsed = int(time.time() - self.start_time)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] (+{elapsed}s) {message}")
    
    def load_lfw_dataset(self):
        """Load LFW dataset from archive directory."""
        self.log_message("Loading LFW dataset from archive...")
        
        # Path to the LFW dataset
        lfw_dir = "archive/lfw-deepfunneled/lfw-deepfunneled"
        
        if not os.path.exists(lfw_dir):
            raise FileNotFoundError(f"LFW dataset not found at {lfw_dir}")
        
        faces = []
        labels = []
        person_names = []
        
        # Get all person directories
        person_dirs = [d for d in os.listdir(lfw_dir) 
                      if os.path.isdir(os.path.join(lfw_dir, d))]
        person_dirs = sorted(person_dirs)
        
        # Select people with enough images
        selected_people = []
        for person in person_dirs:
            person_path = os.path.join(lfw_dir, person)
            images = [f for f in os.listdir(person_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) >= 15:  # At least 15 images
                selected_people.append(person)
            if len(selected_people) >= 8:  # Limit to 8 people for manageable training
                break
        
        if len(selected_people) < 2:
            raise ValueError("Not enough people with sufficient images")
        
        self.log_message(f"Selected {len(selected_people)} people for training")
        
        # Load images for selected people
        for person_idx, person in enumerate(selected_people):
            person_path = os.path.join(lfw_dir, person)
            images = [f for f in os.listdir(person_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            images = sorted(images)[:20]  # Take first 20 images
            
            self.log_message(f"Loading {person}: {len(images)} images")
            
            for img_file in images:
                img_path = os.path.join(person_path, img_file)
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((32, 32))  # Resize to 32x32 for CNN
                    img_array = np.array(img).transpose(2, 0, 1) / 255.0  # CHW format, normalize
                    
                    faces.append(img_array)
                    labels.append(person_idx)
                    
                    if person not in person_names:
                        person_names.append(person)
                        
                except Exception as e:
                    self.log_message(f"Error loading {img_path}: {e}")
                    continue
        
        faces = np.array(faces)
        labels = np.array(labels)
        
        self.log_message(f"✅ LFW dataset loaded: {len(faces)} images, {len(person_names)} people")
        return faces, labels, person_names
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val, 
                   optimizer_name="adam", epochs=15, batch_size=8):
        """Train CNN model with stable implementation."""
        
        self.log_message(f"Training CNN: {model_name} with {optimizer_name}")
        self.log_message(f"Configuration: {epochs} epochs, batch size {batch_size}")
        
        # Create model and optimizer
        num_classes = len(np.unique(y_train))
        model = CNN(num_classes)
        
        if optimizer_name == "adam":
            optimizer = Optimizer(learning_rate=0.0005, optimizer_type='adam')
        else:
            optimizer = Optimizer(learning_rate=0.005, optimizer_type='sgd')
        
        loss_fn = CrossEntropyLoss()
        
        # Training history
        history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'model_name': model_name,
            'optimizer': optimizer_name
        }
        
        self.log_message(f"Starting training with {len(X_train)} samples")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            model.training = True
            train_losses = []
            train_preds = []
            train_true = []
            
            # Shuffle data
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                # Forward pass
                predictions = model.forward(X_batch)
                loss = loss_fn.forward(predictions, y_batch)
                train_losses.append(loss)
                
                # Backward pass
                grad_output = loss_fn.backward(predictions, y_batch)
                model.backward(grad_output)
                
                # Update weights
                optimizer.update(model.get_parameters(), model.get_gradients())
                
                # Track predictions
                pred_classes = np.argmax(predictions, axis=1)
                train_preds.extend(pred_classes)
                train_true.extend(y_batch)
            
            # Validation phase
            model.training = False
            val_losses = []
            val_preds = []
            val_true = []
            
            for i in range(0, len(X_val), batch_size):
                X_batch = X_val[i:i+batch_size]
                y_batch = y_val[i:i+batch_size]
                
                predictions = model.forward(X_batch)
                loss = loss_fn.forward(predictions, y_batch)
                val_losses.append(loss)
                
                pred_classes = np.argmax(predictions, axis=1)
                val_preds.extend(pred_classes)
                val_true.extend(y_batch)
            
            # Calculate metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_acc = accuracy_score(np.array(train_true), np.array(train_preds))
            val_acc = accuracy_score(np.array(val_true), np.array(val_preds))
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            history['epochs'].append(epoch + 1)
            history['train_loss'].append(float(train_loss))
            history['train_acc'].append(float(train_acc))
            history['val_loss'].append(float(val_loss))
            history['val_acc'].append(float(val_acc))
            
            # Log progress
            if epoch % 3 == 0 or epoch < 3:
                self.log_message(f"{model_name} Epoch {epoch+1:2d}/{epochs} [{epoch_time:4.1f}s] - "
                               f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                               f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
        self.log_message(f"✅ {model_name} training completed! Best val accuracy: {best_val_acc:.4f}")
        
        return model, history
    
    def evaluate_model(self, model, model_name, X_test, y_test, class_names):
        """Evaluate model performance."""
        
        self.log_message(f"Evaluating {model_name}...")
        
        model.training = False
        predictions = []
        
        batch_size = 8
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]
            pred_probs = model.forward(X_batch)
            pred_classes = np.argmax(pred_probs, axis=1)
            predictions.extend(pred_classes)
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        accuracy = accuracy_score(np.array(y_test), predictions)
        cm = confusion_matrix(np.array(y_test), predictions)
        
        self.log_message(f"✅ {model_name} Test Accuracy: {accuracy:.4f}")
        
        return {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist()
        }
    
    def run_training(self):
        """Run complete CNN training pipeline."""
        
        print("=" * 60)
        print("CNN FACE RECOGNITION TRAINING")
        print("=" * 60)
        print("This script trains a CNN for face recognition with:")
        print("• Stable gradient computation and backpropagation")
        print("• Multiple optimizers (Adam, SGD)")
        print("• Numerical stability and gradient clipping")
        print("• LFW dataset from archive/lfw-deepfunneled/")
        print("Expected duration: 15-20 minutes")
        print("=" * 60)
        print()
        
        self.log_message("Starting CNN training session")
        
        # Load dataset
        X, y, class_names = self.load_lfw_dataset()
        
        # Split dataset
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        X_train, y_train = X[indices[:train_end]], y[indices[:train_end]]
        X_val, y_val = X[indices[train_end:val_end]], y[indices[train_end:val_end]]
        X_test, y_test = X[indices[val_end:]], y[indices[val_end:]]
        
        self.log_message(f"Dataset split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train models with different optimizers
        optimizers = ['adam', 'sgd']
        
        for i, optimizer_name in enumerate(optimizers):
            model_name = f"CNN_{optimizer_name}"
            
            self.log_message(f"Training model {i+1}/{len(optimizers)}: {model_name}")
            
            # Train model
            model, history = self.train_model(
                model_name, X_train, y_train, X_val, y_val,
                optimizer_name=optimizer_name, epochs=12, batch_size=8
            )
            
            # Evaluate model
            evaluation = self.evaluate_model(
                model, model_name, X_test, y_test, class_names
            )
            
            # Store results
            self.all_results[model_name] = {
                'history': history,
                'evaluation': evaluation
            }
            
            # Save results
            with open(f"{self.results_dir}/{model_name}_history.json", 'w') as f:
                json.dump(history, f, indent=2)
            
            with open(f"{self.results_dir}/{model_name}_evaluation.json", 'w') as f:
                json.dump(evaluation, f, indent=2)
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive training report."""
        
        total_time = time.time() - self.start_time
        
        report = []
        report.append("=" * 60)
        report.append("CNN FACE RECOGNITION TRAINING RESULTS")
        report.append("=" * 60)
        report.append(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total training time: {total_time/60:.1f} minutes")
        report.append("")
        
        report.append("📊 TRAINING RESULTS:")
        for model_name, results in self.all_results.items():
            history = results['history']
            evaluation = results['evaluation']
            
            best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
            test_acc = evaluation['accuracy']
            
            report.append(f"   {model_name}:")
            report.append(f"     ├─ Training epochs: {len(history['epochs'])}")
            report.append(f"     ├─ Best validation accuracy: {best_val_acc:.4f}")
            report.append(f"     ├─ Final test accuracy: {test_acc:.4f}")
            report.append(f"     └─ Training time: ~{total_time/(60*len(self.all_results)):.1f} min")
        
        report.append("")
        report.append("✅ CNN IMPLEMENTATION FEATURES:")
        report.append("   ✅ Gradient computation and backpropagation")
        report.append("   ✅ Stable numerical implementation")
        report.append("   ✅ Gradient clipping for stability")
        report.append("   ✅ Multiple optimizers (Adam, SGD)")
        report.append("   ✅ LFW face recognition dataset training")
        report.append("   ✅ Comprehensive evaluation and reporting")
        
        report.append("")
        report.append("🎉 TRAINING COMPLETE!")
        report.append(f"Total time: {total_time/60:.1f} minutes")
        report.append("Check training_results/ directory for detailed results")
        report.append("=" * 60)
        
        # Save and display report
        report_text = "\n".join(report)
        
        with open(f"{self.results_dir}/TRAINING_REPORT.txt", 'w') as f:
            f.write(report_text)
        
        self.log_message("📋 Training report generated")
        print("\n" + report_text)


def main():
    """Main function to run CNN training."""
    
    print("CNN Face Recognition Training")
    print("============================")
    print("This will train a CNN for face recognition using the LFW dataset.")
    print("Dataset location: archive/lfw-deepfunneled/lfw-deepfunneled/")
    print("Training includes proper gradient computation with numerical stability.")
    print("Expected duration: 15-20 minutes")
    print()
    
    response = input("Do you want to start training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Start training
    trainer = CNNTrainer()
    trainer.run_training()


if __name__ == "__main__":
    main() 