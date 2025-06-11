"""
Classification evaluation metrics and tools
"""

import numpy as np
from .confusion_matrix import ConfusionMatrix


class ClassificationEvaluator:
    """Comprehensive evaluation for classification tasks"""
    
    def __init__(self, y_true, y_pred, y_pred_proba=None, class_names=None):
        """
        Initialize classification evaluator
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            class_names: Names of classes (optional)
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_pred_proba = np.asarray(y_pred_proba) if y_pred_proba is not None else None
        self.class_names = class_names
        
        # Create confusion matrix
        self.confusion_matrix = ConfusionMatrix(y_true, y_pred, class_names)
    
    def accuracy(self):
        """Overall accuracy"""
        return self.confusion_matrix.accuracy()
    
    def precision(self, average='macro'):
        """Precision score"""
        return self.confusion_matrix.precision(average=average)
    
    def recall(self, average='macro'):
        """Recall score"""
        return self.confusion_matrix.recall(average=average)
    
    def f1_score(self, average='macro'):
        """F1 score"""
        return self.confusion_matrix.f1_score(average=average)
    
    def specificity(self, average='macro'):
        """Specificity score"""
        return self.confusion_matrix.specificity(average=average)
    
    def balanced_accuracy(self):
        """Balanced accuracy (mean of per-class recalls)"""
        return self.recall(average='macro')
    
    def matthews_correlation_coefficient(self):
        """Matthews Correlation Coefficient (for binary classification)"""
        if len(self.confusion_matrix.classes) != 2:
            raise ValueError("MCC is only defined for binary classification")
        
        cm = self.confusion_matrix.matrix
        tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
        
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def cohen_kappa(self):
        """Cohen's Kappa coefficient"""
        cm = self.confusion_matrix.matrix
        n = np.sum(cm)
        
        # Observed agreement
        po = np.trace(cm) / n
        
        # Expected agreement
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (n ** 2)
        
        return (po - pe) / (1 - pe) if pe != 1 else 0.0
    
    def classification_report(self):
        """Generate comprehensive classification report"""
        return self.confusion_matrix.classification_report()
    
    def plot_confusion_matrix(self, normalize=False, save_path=None):
        """Plot confusion matrix"""
        self.confusion_matrix.plot(normalize=normalize, save_path=save_path)
    
    def roc_auc_score(self):
        """ROC AUC score (requires predicted probabilities)"""
        if self.y_pred_proba is None:
            raise ValueError("ROC AUC requires predicted probabilities")
        
        from sklearn.metrics import roc_auc_score
        
        if len(self.confusion_matrix.classes) == 2:
            # Binary classification
            return roc_auc_score(self.y_true, self.y_pred_proba[:, 1])
        else:
            # Multi-class classification
            return roc_auc_score(self.y_true, self.y_pred_proba, multi_class='ovr')
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve (binary classification only)"""
        if self.y_pred_proba is None:
            raise ValueError("ROC curve requires predicted probabilities")
        
        if len(self.confusion_matrix.classes) != 2:
            raise ValueError("ROC curve plotting is only supported for binary classification")
        
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_metrics_dict(self):
        """Get all metrics as dictionary"""
        metrics = {
            'accuracy': self.accuracy(),
            'precision_macro': self.precision('macro'),
            'precision_micro': self.precision('micro'),
            'precision_weighted': self.precision('weighted'),
            'recall_macro': self.recall('macro'),
            'recall_micro': self.recall('micro'),
            'recall_weighted': self.recall('weighted'),
            'f1_macro': self.f1_score('macro'),
            'f1_micro': self.f1_score('micro'),
            'f1_weighted': self.f1_score('weighted'),
            'balanced_accuracy': self.balanced_accuracy(),
            'cohen_kappa': self.cohen_kappa()
        }
        
        # Add binary classification specific metrics
        if len(self.confusion_matrix.classes) == 2:
            metrics['mcc'] = self.matthews_correlation_coefficient()
            
            if self.y_pred_proba is not None:
                try:
                    metrics['roc_auc'] = self.roc_auc_score()
                except ImportError:
                    pass  # sklearn not available
        
        return metrics 