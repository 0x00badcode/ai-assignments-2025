"""Visualization utilities for CNN training and evaluation."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional, Tuple, Any
import seaborn as sns


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 4)) -> None:
    """Plot training history including loss and metrics.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    metrics = list(history.keys())
    n_metrics = len(metrics)
    
    if n_metrics == 0:
        print("No metrics to plot")
        return
    
    # Determine subplot layout
    if n_metrics == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]
    elif n_metrics == 2:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i] if i < len(axes) else axes[-1]
        
        epochs = range(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], 'b-', label=f'Training {metric}', linewidth=2)
        
        # Check for validation metric
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(epochs, history[val_metric], 'r--', label=f'Validation {metric}', linewidth=2)
        
        ax.set_title(f'{metric.capitalize()} Over Time')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: Optional[List[str]] = None,
                         normalize: bool = False,
                         title: str = 'Confusion Matrix',
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (8, 6)) -> None:
    """Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: Names of the classes
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized' if normalize else 'Count'})
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_filters(weights: np.ndarray, 
                layer_name: str = 'Conv Layer',
                max_filters: int = 64,
                save_path: Optional[str] = None,
                figsize: Tuple[int, int] = (12, 8)) -> None:
    """Visualize convolutional filters.
    
    Args:
        weights: Filter weights with shape (out_channels, in_channels, height, width)
        layer_name: Name of the layer for the title
        max_filters: Maximum number of filters to display
        save_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    if len(weights.shape) != 4:
        print(f"Expected 4D weights, got {weights.shape}")
        return
    
    out_channels, in_channels, h, w = weights.shape
    n_filters = min(max_filters, out_channels)
    
    # Calculate grid size
    n_cols = int(np.ceil(np.sqrt(n_filters)))
    n_rows = int(np.ceil(n_filters / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(n_filters):
        ax = axes[i]
        
        # Get filter weights for first input channel or average across channels
        if in_channels == 1:
            filter_img = weights[i, 0]
        elif in_channels == 3:
            # For RGB, show as color image
            filter_img = weights[i].transpose(1, 2, 0)
            # Normalize to [0, 1]
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
        else:
            # Average across input channels
            filter_img = np.mean(weights[i], axis=0)
        
        if len(filter_img.shape) == 2:
            ax.imshow(filter_img, cmap='viridis')
        else:
            ax.imshow(filter_img)
        
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{layer_name} Filters', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Filter visualization saved to {save_path}")
    
    plt.show()


def plot_feature_maps(feature_maps: np.ndarray, 
                     layer_name: str = 'Feature Maps',
                     max_maps: int = 16,
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (12, 8)) -> None:
    """Visualize feature maps from a convolutional layer.
    
    Args:
        feature_maps: Feature maps with shape (channels, height, width)
        layer_name: Name of the layer for the title
        max_maps: Maximum number of feature maps to display
        save_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    if len(feature_maps.shape) != 3:
        print(f"Expected 3D feature maps, got {feature_maps.shape}")
        return
    
    channels, h, w = feature_maps.shape
    n_maps = min(max_maps, channels)
    
    # Calculate grid size
    n_cols = int(np.ceil(np.sqrt(n_maps)))
    n_rows = int(np.ceil(n_maps / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(n_maps):
        ax = axes[i]
        ax.imshow(feature_maps[i], cmap='viridis')
        ax.set_title(f'Map {i+1}')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_maps, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{layer_name} Feature Maps', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature maps saved to {save_path}")
    
    plt.show()


def plot_loss_landscape(loss_values: np.ndarray,
                       param1_range: np.ndarray,
                       param2_range: np.ndarray,
                       param1_name: str = 'Parameter 1',
                       param2_name: str = 'Parameter 2',
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 8)) -> None:
    """Plot 2D loss landscape.
    
    Args:
        loss_values: 2D array of loss values
        param1_range: Range of first parameter
        param2_range: Range of second parameter
        param1_name: Name of first parameter
        param2_name: Name of second parameter
        save_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Create contour plot
    contour = plt.contourf(param1_range, param2_range, loss_values, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Loss')
    
    # Add contour lines
    plt.contour(param1_range, param2_range, loss_values, levels=20, colors='white', alpha=0.5, linewidths=0.5)
    
    plt.xlabel(param1_name)
    plt.ylabel(param2_name)
    plt.title('Loss Landscape')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss landscape saved to {save_path}")
    
    plt.show()


def plot_learning_curves(train_sizes: np.ndarray,
                        train_scores: np.ndarray,
                        val_scores: np.ndarray,
                        metric_name: str = 'Score',
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 6)) -> None:
    """Plot learning curves showing performance vs training set size.
    
    Args:
        train_sizes: Array of training set sizes
        train_scores: Training scores for each size
        val_scores: Validation scores for each size
        metric_name: Name of the metric being plotted
        save_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot curves
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label=f'Training {metric_name}')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label=f'Validation {metric_name}')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel(metric_name)
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to {save_path}")
    
    plt.show()


def plot_predictions_vs_actual(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              title: str = 'Predictions vs Actual',
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (8, 8)) -> None:
    """Plot predictions vs actual values for regression tasks.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions vs actual plot saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, 
                  y_scores: np.ndarray,
                  title: str = 'ROC Curve',
                  save_path: Optional[str] = None,
                  figsize: Tuple[int, int] = (8, 6)) -> None:
    """Plot ROC curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    # Calculate ROC curve
    thresholds = np.linspace(1, 0, 100)
    tpr_values = []
    fpr_values = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    # Calculate AUC
    auc = np.trapz(tpr_values, fpr_values)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr_values, tpr_values, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def plot_sample_images(images: np.ndarray, 
                      labels: Optional[np.ndarray] = None,
                      predictions: Optional[np.ndarray] = None,
                      class_names: Optional[List[str]] = None,
                      n_samples: int = 25,
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (12, 12)) -> None:
    """Plot sample images with labels and predictions.
    
    Args:
        images: Images to display
        labels: True labels
        predictions: Predicted labels
        class_names: Names of the classes
        n_samples: Number of samples to display
        save_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    n_samples = min(n_samples, len(images))
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(n_samples):
        ax = axes[i]
        
        # Get image
        img = images[i]
        if len(img.shape) == 3:
            if img.shape[0] in [1, 3]:  # CHW format
                img = img.transpose(1, 2, 0)
            if img.shape[2] == 1:  # Grayscale
                img = img.squeeze(2)
        
        # Display image
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        
        # Create title
        title_parts = []
        if labels is not None:
            true_label = labels[i] if labels[i].ndim == 0 else np.argmax(labels[i])
            if class_names:
                title_parts.append(f"True: {class_names[true_label]}")
            else:
                title_parts.append(f"True: {true_label}")
        
        if predictions is not None:
            pred_label = predictions[i] if predictions[i].ndim == 0 else np.argmax(predictions[i])
            if class_names:
                title_parts.append(f"Pred: {class_names[pred_label]}")
            else:
                title_parts.append(f"Pred: {pred_label}")
        
        if title_parts:
            ax.set_title('\n'.join(title_parts), fontsize=8)
        
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample images saved to {save_path}")
    
    plt.show() 