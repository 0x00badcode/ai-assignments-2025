"""
Regression evaluation metrics and tools
"""

import numpy as np
import matplotlib.pyplot as plt


class RegressionEvaluator:
    """Comprehensive evaluation for regression tasks"""
    
    def __init__(self, y_true, y_pred):
        """
        Initialize regression evaluator
        
        Args:
            y_true: True values
            y_pred: Predicted values
        """
        self.y_true = np.asarray(y_true).flatten()
        self.y_pred = np.asarray(y_pred).flatten()
        
        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length")
    
    def mean_squared_error(self):
        """Mean Squared Error"""
        return np.mean((self.y_true - self.y_pred) ** 2)
    
    def root_mean_squared_error(self):
        """Root Mean Squared Error"""
        return np.sqrt(self.mean_squared_error())
    
    def mean_absolute_error(self):
        """Mean Absolute Error"""
        return np.mean(np.abs(self.y_true - self.y_pred))
    
    def mean_absolute_percentage_error(self):
        """Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = self.y_true != 0
        if not np.any(mask):
            return np.inf
        return np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100
    
    def r_squared(self):
        """R-squared (coefficient of determination)"""
        ss_res = np.sum((self.y_true - self.y_pred) ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def adjusted_r_squared(self, n_features):
        """Adjusted R-squared"""
        n = len(self.y_true)
        r2 = self.r_squared()
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    
    def mean_squared_log_error(self):
        """Mean Squared Logarithmic Error"""
        # Only for non-negative values
        if np.any(self.y_true < 0) or np.any(self.y_pred < 0):
            raise ValueError("MSLE requires non-negative values")
        return np.mean((np.log1p(self.y_true) - np.log1p(self.y_pred)) ** 2)
    
    def max_error(self):
        """Maximum residual error"""
        return np.max(np.abs(self.y_true - self.y_pred))
    
    def explained_variance_score(self):
        """Explained variance score"""
        y_diff_avg = np.mean(self.y_true - self.y_pred)
        numerator = np.mean((self.y_true - self.y_pred - y_diff_avg) ** 2)
        denominator = np.var(self.y_true)
        return 1 - (numerator / denominator) if denominator != 0 else 0.0
    
    def median_absolute_error(self):
        """Median Absolute Error"""
        return np.median(np.abs(self.y_true - self.y_pred))
    
    def mean_absolute_scaled_error(self, y_train):
        """Mean Absolute Scaled Error (requires training data)"""
        y_train = np.asarray(y_train).flatten()
        # Calculate naive forecast error (using previous value)
        naive_error = np.mean(np.abs(y_train[1:] - y_train[:-1]))
        mae = self.mean_absolute_error()
        return mae / naive_error if naive_error != 0 else np.inf
    
    def residuals(self):
        """Calculate residuals"""
        return self.y_true - self.y_pred
    
    def plot_predictions(self, save_path=None, figsize=(12, 4)):
        """
        Plot predictions vs actual values
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Scatter plot: Predicted vs Actual
        axes[0].scatter(self.y_true, self.y_pred, alpha=0.6)
        min_val = min(np.min(self.y_true), np.min(self.y_pred))
        max_val = max(np.max(self.y_true), np.max(self.y_pred))
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Predicted vs Actual')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = self.residuals()
        axes[1].scatter(self.y_pred, residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals vs Predicted')
        axes[1].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[2].axvline(x=0, color='r', linestyle='--')
        axes[2].set_xlabel('Residuals')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Residuals Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_residuals_vs_features(self, X, feature_names=None, save_path=None):
        """
        Plot residuals vs features
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            save_path: Path to save the plot
        """
        X = np.asarray(X)
        residuals = self.residuals()
        n_features = X.shape[1]
        
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(n_features)]
        
        # Calculate subplot layout
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(n_features):
            axes[i].scatter(X[:, i], residuals, alpha=0.6)
            axes[i].axhline(y=0, color='r', linestyle='--')
            axes[i].set_xlabel(feature_names[i])
            axes[i].set_ylabel('Residuals')
            axes[i].set_title(f'Residuals vs {feature_names[i]}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_metrics_dict(self, n_features=None, y_train=None):
        """
        Get all metrics as dictionary
        
        Args:
            n_features: Number of features (for adjusted R²)
            y_train: Training data (for MASE)
        """
        metrics = {
            'mse': self.mean_squared_error(),
            'rmse': self.root_mean_squared_error(),
            'mae': self.mean_absolute_error(),
            'mape': self.mean_absolute_percentage_error(),
            'r2': self.r_squared(),
            'explained_variance': self.explained_variance_score(),
            'median_ae': self.median_absolute_error(),
            'max_error': self.max_error()
        }
        
        # Optional metrics
        if n_features is not None:
            metrics['adjusted_r2'] = self.adjusted_r_squared(n_features)
        
        if y_train is not None:
            try:
                metrics['mase'] = self.mean_absolute_scaled_error(y_train)
            except:
                pass  # Skip if calculation fails
        
        # MSLE only for non-negative values
        try:
            metrics['msle'] = self.mean_squared_log_error()
        except ValueError:
            pass  # Skip if negative values present
        
        return metrics
    
    def regression_report(self, n_features=None, y_train=None):
        """Generate comprehensive regression report"""
        metrics = self.get_metrics_dict(n_features, y_train)
        
        report = "Regression Evaluation Report\n"
        report += "=" * 40 + "\n"
        
        report += f"Mean Squared Error (MSE):     {metrics['mse']:.6f}\n"
        report += f"Root Mean Squared Error:      {metrics['rmse']:.6f}\n"
        report += f"Mean Absolute Error (MAE):    {metrics['mae']:.6f}\n"
        report += f"Mean Absolute % Error:        {metrics['mape']:.2f}%\n"
        report += f"R-squared:                    {metrics['r2']:.6f}\n"
        
        if 'adjusted_r2' in metrics:
            report += f"Adjusted R-squared:           {metrics['adjusted_r2']:.6f}\n"
        
        report += f"Explained Variance:           {metrics['explained_variance']:.6f}\n"
        report += f"Median Absolute Error:        {metrics['median_ae']:.6f}\n"
        report += f"Max Error:                    {metrics['max_error']:.6f}\n"
        
        if 'msle' in metrics:
            report += f"Mean Squared Log Error:       {metrics['msle']:.6f}\n"
        
        if 'mase' in metrics:
            report += f"Mean Absolute Scaled Error:   {metrics['mase']:.6f}\n"
        
        return report 