"""Evaluation metrics for CNN models."""

import numpy as np
from typing import Optional, List, Dict, Any, Union


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    return np.mean(y_true == y_pred)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> Union[float, np.ndarray]:
    """Calculate precision score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', None)
        
    Returns:
        Precision score(s)
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    classes = np.unique(y_true)
    precisions = []
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        precisions.append(precision)
    
    precisions = np.array(precisions)
    
    if average == 'macro':
        return np.mean(precisions)
    elif average == 'micro':
        tp_total = np.sum([np.sum((y_true == cls) & (y_pred == cls)) for cls in classes])
        fp_total = np.sum([np.sum((y_true != cls) & (y_pred == cls)) for cls in classes])
        return tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0
    elif average == 'weighted':
        weights = [np.sum(y_true == cls) for cls in classes]
        return np.average(precisions, weights=weights)
    else:
        return precisions


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> Union[float, np.ndarray]:
    """Calculate recall score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', None)
        
    Returns:
        Recall score(s)
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    classes = np.unique(y_true)
    recalls = []
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        recalls.append(recall)
    
    recalls = np.array(recalls)
    
    if average == 'macro':
        return np.mean(recalls)
    elif average == 'micro':
        tp_total = np.sum([np.sum((y_true == cls) & (y_pred == cls)) for cls in classes])
        fn_total = np.sum([np.sum((y_true == cls) & (y_pred != cls)) for cls in classes])
        return tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0
    elif average == 'weighted':
        weights = [np.sum(y_true == cls) for cls in classes]
        return np.average(recalls, weights=weights)
    else:
        return recalls


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> Union[float, np.ndarray]:
    """Calculate F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', None)
        
    Returns:
        F1 score(s)
    """
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    
    if isinstance(precision, np.ndarray):
        f1 = np.zeros_like(precision)
        mask = (precision + recall) > 0
        f1[mask] = 2 * precision[mask] * recall[mask] / (precision[mask] + recall[mask])
        return f1
    else:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                    labels: Optional[List[int]] = None) -> np.ndarray:
    """Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of labels to index the matrix
        
    Returns:
        Confusion matrix
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                         target_names: Optional[List[str]] = None,
                         digits: int = 2) -> str:
    """Build a text report showing the main classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of the target classes
        digits: Number of digits for formatting output
        
    Returns:
        Text report
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    labels = np.unique(y_true)
    
    if target_names is None:
        target_names = [f'class_{i}' for i in labels]
    
    # Calculate metrics for each class
    precisions = precision_score(y_true, y_pred, average=None)
    recalls = recall_score(y_true, y_pred, average=None)
    f1s = f1_score(y_true, y_pred, average=None)
    supports = [np.sum(y_true == label) for label in labels]
    
    # Header
    name_width = max(len(name) for name in target_names)
    width = max(name_width, digits + 5, 7)
    
    headers = ["precision", "recall", "f1-score", "support"]
    fmt = f"{{:>{width}s}} " + " ".join([f"{{:>{width}}}"] * len(headers))
    
    report = fmt.format("", *headers) + "\n\n"
    
    # Per-class metrics
    for i, label in enumerate(labels):
        values = [
            f"{precisions[i]:.{digits}f}",
            f"{recalls[i]:.{digits}f}",
            f"{f1s[i]:.{digits}f}",
            f"{supports[i]}"
        ]
        report += fmt.format(target_names[i], *values) + "\n"
    
    # Averages
    report += "\n"
    
    # Macro average
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    total_support = sum(supports)
    
    values = [
        f"{macro_precision:.{digits}f}",
        f"{macro_recall:.{digits}f}",
        f"{macro_f1:.{digits}f}",
        f"{total_support}"
    ]
    report += fmt.format("macro avg", *values) + "\n"
    
    # Weighted average
    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    values = [
        f"{weighted_precision:.{digits}f}",
        f"{weighted_recall:.{digits}f}",
        f"{weighted_f1:.{digits}f}",
        f"{total_support}"
    ]
    report += fmt.format("weighted avg", *values) + "\n"
    
    return report


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Mean absolute error
    """
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² score for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def top_k_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Calculate top-k accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
    
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)


def face_verification_metrics(distances: np.ndarray, labels: np.ndarray, 
                            threshold: float = 0.5) -> Dict[str, float]:
    """Calculate face verification metrics.
    
    Args:
        distances: Distances between face pairs
        labels: True labels (1 for same person, 0 for different)
        threshold: Distance threshold for verification
        
    Returns:
        Dictionary of metrics
    """
    # Predictions: 1 if distance < threshold (same person), 0 otherwise
    predictions = (distances < threshold).astype(int)
    
    # Calculate metrics
    tp = np.sum((labels == 1) & (predictions == 1))
    tn = np.sum((labels == 0) & (predictions == 0))
    fp = np.sum((labels == 0) & (predictions == 1))
    fn = np.sum((labels == 1) & (predictions == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # False acceptance rate and false rejection rate
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_acceptance_rate': far,
        'false_rejection_rate': frr,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    } 