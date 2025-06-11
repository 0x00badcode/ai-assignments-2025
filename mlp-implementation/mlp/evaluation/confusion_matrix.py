"""
Confusion Matrix implementation for classification evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ConfusionMatrix:
    """Confusion Matrix for classification evaluation"""
    
    def __init__(self, y_true, y_pred, class_names=None):
        """
        Initialize confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes (optional)
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        
        # Get unique classes
        self.classes = np.unique(np.concatenate([self.y_true, self.y_pred]))
        self.n_classes = len(self.classes)
        
        if class_names is None:
            self.class_names = [f"Class {i}" for i in self.classes]
        else:
            self.class_names = class_names
        
        # Compute confusion matrix
        self.matrix = self._compute_matrix()
    
    def _compute_matrix(self):
        """Compute the confusion matrix"""
        matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)
        
        for i, true_class in enumerate(self.classes):
            for j, pred_class in enumerate(self.classes):
                matrix[i, j] = np.sum((self.y_true == true_class) & (self.y_pred == pred_class))
        
        return matrix
    
    def accuracy(self):
        """Compute overall accuracy"""
        return np.trace(self.matrix) / np.sum(self.matrix)
    
    def precision(self, class_idx=None, average='macro'):
        """
        Compute precision
        
        Args:
            class_idx: Specific class index (None for all classes)
            average: 'macro', 'micro', or 'weighted'
        """
        if class_idx is not None:
            tp = self.matrix[class_idx, class_idx]
            fp = np.sum(self.matrix[:, class_idx]) - tp
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        precisions = []
        for i in range(self.n_classes):
            precisions.append(self.precision(i))
        
        if average == 'macro':
            return np.mean(precisions)
        elif average == 'micro':
            tp_total = np.trace(self.matrix)
            fp_total = np.sum(self.matrix) - tp_total
            return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        elif average == 'weighted':
            weights = np.sum(self.matrix, axis=1)
            return np.average(precisions, weights=weights)
        else:
            return precisions
    
    def recall(self, class_idx=None, average='macro'):
        """
        Compute recall (sensitivity)
        
        Args:
            class_idx: Specific class index (None for all classes)
            average: 'macro', 'micro', or 'weighted'
        """
        if class_idx is not None:
            tp = self.matrix[class_idx, class_idx]
            fn = np.sum(self.matrix[class_idx, :]) - tp
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        recalls = []
        for i in range(self.n_classes):
            recalls.append(self.recall(i))
        
        if average == 'macro':
            return np.mean(recalls)
        elif average == 'micro':
            tp_total = np.trace(self.matrix)
            fn_total = np.sum(self.matrix) - tp_total
            return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        elif average == 'weighted':
            weights = np.sum(self.matrix, axis=1)
            return np.average(recalls, weights=weights)
        else:
            return recalls
    
    def f1_score(self, class_idx=None, average='macro'):
        """
        Compute F1-score
        
        Args:
            class_idx: Specific class index (None for all classes)
            average: 'macro', 'micro', or 'weighted'
        """
        if class_idx is not None:
            p = self.precision(class_idx)
            r = self.recall(class_idx)
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        if average == 'micro':
            p = self.precision(average='micro')
            r = self.recall(average='micro')
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        f1_scores = []
        for i in range(self.n_classes):
            f1_scores.append(self.f1_score(i))
        
        if average == 'macro':
            return np.mean(f1_scores)
        elif average == 'weighted':
            weights = np.sum(self.matrix, axis=1)
            return np.average(f1_scores, weights=weights)
        else:
            return f1_scores
    
    def specificity(self, class_idx=None, average='macro'):
        """
        Compute specificity (true negative rate)
        
        Args:
            class_idx: Specific class index (None for all classes)
            average: 'macro', 'micro', or 'weighted'
        """
        if class_idx is not None:
            tn = np.sum(self.matrix) - np.sum(self.matrix[class_idx, :]) - np.sum(self.matrix[:, class_idx]) + self.matrix[class_idx, class_idx]
            fp = np.sum(self.matrix[:, class_idx]) - self.matrix[class_idx, class_idx]
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        specificities = []
        for i in range(self.n_classes):
            specificities.append(self.specificity(i))
        
        if average == 'macro':
            return np.mean(specificities)
        elif average == 'weighted':
            weights = np.sum(self.matrix, axis=1)
            return np.average(specificities, weights=weights)
        else:
            return specificities
    
    def plot(self, normalize=False, title="Confusion Matrix", figsize=(8, 6), save_path=None):
        """
        Plot confusion matrix
        
        Args:
            normalize: Whether to normalize the matrix
            title: Title for the plot
            figsize: Figure size
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=figsize)
        
        matrix_to_plot = self.matrix.copy()
        if normalize:
            matrix_to_plot = matrix_to_plot.astype('float') / matrix_to_plot.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        sns.heatmap(matrix_to_plot, 
                   annot=True, 
                   fmt=fmt,
                   cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def classification_report(self):
        """Generate a comprehensive classification report"""
        report = f"Classification Report\n"
        report += f"{'='*50}\n"
        report += f"Overall Accuracy: {self.accuracy():.4f}\n\n"
        
        # Per-class metrics
        report += f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n"
        report += f"{'-'*65}\n"
        
        for i, class_name in enumerate(self.class_names):
            precision = self.precision(i)
            recall = self.recall(i)
            f1 = self.f1_score(i)
            support = np.sum(self.matrix[i, :])
            
            report += f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}\n"
        
        # Macro averages
        report += f"\n{'Macro Avg':<15} {self.precision(average='macro'):<10.4f} {self.recall(average='macro'):<10.4f} {self.f1_score(average='macro'):<10.4f} {np.sum(self.matrix):<10}\n"
        report += f"{'Weighted Avg':<15} {self.precision(average='weighted'):<10.4f} {self.recall(average='weighted'):<10.4f} {self.f1_score(average='weighted'):<10.4f} {np.sum(self.matrix):<10}\n"
        
        return report
    
    def __str__(self):
        """String representation of confusion matrix"""
        return f"Confusion Matrix ({self.n_classes}x{self.n_classes}):\n{self.matrix}"
    
    def __repr__(self):
        return self.__str__() 