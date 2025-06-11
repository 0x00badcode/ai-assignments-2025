"""
Boston Housing Regression Example
Demonstrates the MLP library on the Boston Housing regression task
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from sklearn.datasets import fetch_california_housing
    DATASET_AVAILABLE = 'california'
except ImportError:
    DATASET_AVAILABLE = None
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add parent directory to path to import mlp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp import MLP, StoppingCriteria
from mlp.optimizers import Adam, SGD, RMSProp
from mlp.regularization import L1Regularization, L2Regularization, ElasticNetRegularization
from mlp.evaluation import RegressionEvaluator


def load_boston_data():
    """Load and preprocess housing data"""
    
    if DATASET_AVAILABLE == 'california':
        print("Loading California Housing data...")
        california = fetch_california_housing()
        X, y = california.data, california.target
        feature_names = california.feature_names
        print("Using California Housing dataset as replacement for Boston Housing")
    else:
        # Generate synthetic data similar to Boston housing
        print("Generating synthetic housing data...")
        np.random.seed(42)
        n_samples, n_features = 506, 13
        
        X = np.random.randn(n_samples, n_features)
        # Create realistic target values
        y = (X[:, 0] * 3.5 + X[:, 1] * -2.1 + X[:, 2] * 1.8 + 
             X[:, 3] * -0.5 + np.random.randn(n_samples) * 2 + 25)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, feature_names


def main():
    print("Boston Housing Regression with MLP from Scratch")
    print("=" * 50)
    
    # Load data
    X, y, feature_names = load_boston_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Further split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape targets for MLP (need 2D)
    y_train_reshaped = y_train_split.reshape(-1, 1)
    y_val_reshaped = y_val.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Validation set: {X_val_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    print(f"Number of features: {X_train_scaled.shape[1]}")
    
    # Create and configure the model
    print("\nBuilding MLP model...")
    model = MLP(task_type='regression')
    
    # Add layers
    model.add_first_dense(X.shape[1], 64, activation='relu', weight_init='he')
    model.add_dropout(0.2)
    model.add_dense(32, activation='relu', weight_init='he')
    model.add_dropout(0.2)
    model.add_dense(16, activation='relu', weight_init='he')
    model.add_dense(1, activation='linear', weight_init='xavier')
    
    # Compile model with regularization
    regularization = ElasticNetRegularization(lambda_reg=0.001, l1_ratio=0.5)
    
    model.compile(
        optimizer='adam',
        loss='mse',
        regularization=regularization,
        learning_rate=0.001
    )
    
    # Print model summary
    model.summary()
    
    # Set up early stopping
    early_stopping = StoppingCriteria(
        patience=20,
        min_delta=0.001,
        monitor='val_loss',
        mode='min'
    )
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train_scaled, y_train_reshaped,
        epochs=200,
        batch_size=32,
        validation_data=(X_val_scaled, y_val_reshaped),
        verbose=1,
        early_stopping=early_stopping,
        shuffle=True
    )
    
    # Plot training history
    model.plot_history(save_path='boston_training_history.png')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = model.evaluate(X_test_scaled, y_test_reshaped)
    print(f"Test Loss (MSE): {test_metrics['loss']:.4f}")
    print(f"Test RMSE: {np.sqrt(test_metrics['loss']):.4f}")
    
    # Get predictions
    predictions = model.predict(X_test_scaled).flatten()
    
    # Comprehensive evaluation
    print("\nComprehensive evaluation...")
    evaluator = RegressionEvaluator(y_test, predictions)
    
    # Print detailed metrics
    print(evaluator.regression_report(n_features=X.shape[1], y_train=y_train))
    
    # Plot prediction results
    evaluator.plot_predictions(save_path='boston_predictions.png')
    
    # Plot residuals vs features
    evaluator.plot_residuals_vs_features(
        X_test_scaled, feature_names, save_path='boston_residuals_vs_features.png'
    )
    
    # Save the model
    print("\nSaving model...")
    model.save('boston_mlp_model.pkl')
    
    # Compare different regularization techniques
    print("\nComparing regularization techniques...")
    regularizations = [
        ('None', None),
        ('L1', L1Regularization(lambda_reg=0.001)),
        ('L2', L2Regularization(lambda_reg=0.001)),
        ('Elastic Net', ElasticNetRegularization(lambda_reg=0.001, l1_ratio=0.5))
    ]
    
    reg_results = {}
    
    for reg_name, regularization in regularizations:
        print(f"\nTraining with {reg_name} regularization...")
        
        # Create new model
        model_reg = MLP(task_type='regression')
        model_reg.add_first_dense(X.shape[1], 32, activation='relu')
        model_reg.add_dense(16, activation='relu')
        model_reg.add_dense(1, activation='linear')
        
        model_reg.compile(
            optimizer='adam',
            loss='mse',
            regularization=regularization,
            learning_rate=0.001
        )
        
        # Train for fewer epochs for comparison
        history_reg = model_reg.fit(
            X_train_scaled, y_train_reshaped,
            epochs=50,
            batch_size=32,
            validation_data=(X_val_scaled, y_val_reshaped),
            verbose=0
        )
        
        # Evaluate
        test_metrics_reg = model_reg.evaluate(X_test_scaled, y_test_reshaped)
        predictions_reg = model_reg.predict(X_test_scaled).flatten()
        evaluator_reg = RegressionEvaluator(y_test, predictions_reg)
        
        reg_results[reg_name] = {
            'final_loss': history_reg['loss'][-1],
            'test_mse': test_metrics_reg['loss'],
            'test_r2': evaluator_reg.r_squared()
        }
    
    # Plot regularization comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    reg_names = list(reg_results.keys())
    train_losses = [reg_results[name]['final_loss'] for name in reg_names]
    test_mses = [reg_results[name]['test_mse'] for name in reg_names]
    test_r2s = [reg_results[name]['test_r2'] for name in reg_names]
    
    axes[0].bar(reg_names, train_losses)
    axes[0].set_title('Final Training Loss')
    axes[0].set_ylabel('MSE')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(reg_names, test_mses)
    axes[1].set_title('Test MSE')
    axes[1].set_ylabel('MSE')
    axes[1].tick_params(axis='x', rotation=45)
    
    axes[2].bar(reg_names, test_r2s)
    axes[2].set_title('Test R²')
    axes[2].set_ylabel('R²')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('regularization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nRegularization Comparison Results:")
    for reg_name, results in reg_results.items():
        print(f"{reg_name:12} - Train Loss: {results['final_loss']:.3f}, "
              f"Test MSE: {results['test_mse']:.3f}, Test R²: {results['test_r2']:.3f}")
    
    # Demonstrate different optimizers
    print("\nComparing different optimizers...")
    optimizers = [
        ('SGD', SGD(learning_rate=0.01)),
        ('RMSProp', RMSProp(learning_rate=0.001)),
        ('Adam', Adam(learning_rate=0.001))
    ]
    
    optimizer_results = {}
    
    for opt_name, optimizer in optimizers:
        print(f"\nTraining with {opt_name}...")
        
        # Create new model
        model_opt = MLP(task_type='regression')
        model_opt.add_first_dense(X.shape[1], 32, activation='relu')
        model_opt.add_dense(16, activation='relu')
        model_opt.add_dense(1, activation='linear')
        
        model_opt.compile(optimizer=optimizer, loss='mse')
        
        # Train for fewer epochs for comparison
        history_opt = model_opt.fit(
            X_train_scaled, y_train_reshaped,
            epochs=30,
            batch_size=32,
            validation_data=(X_val_scaled, y_val_reshaped),
            verbose=0
        )
        
        # Evaluate
        test_metrics_opt = model_opt.evaluate(X_test_scaled, y_test_reshaped)
        predictions_opt = model_opt.predict(X_test_scaled).flatten()
        evaluator_opt = RegressionEvaluator(y_test, predictions_opt)
        
        optimizer_results[opt_name] = {
            'final_loss': history_opt['loss'][-1],
            'test_mse': test_metrics_opt['loss'],
            'test_r2': evaluator_opt.r_squared()
        }
    
    # Plot optimizer comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    opt_names = list(optimizer_results.keys())
    train_losses_opt = [optimizer_results[name]['final_loss'] for name in opt_names]
    test_mses_opt = [optimizer_results[name]['test_mse'] for name in opt_names]
    test_r2s_opt = [optimizer_results[name]['test_r2'] for name in opt_names]
    
    axes[0].bar(opt_names, train_losses_opt)
    axes[0].set_title('Final Training Loss')
    axes[0].set_ylabel('MSE')
    
    axes[1].bar(opt_names, test_mses_opt)
    axes[1].set_title('Test MSE')
    axes[1].set_ylabel('MSE')
    
    axes[2].bar(opt_names, test_r2s_opt)
    axes[2].set_title('Test R²')
    axes[2].set_ylabel('R²')
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nOptimizer Comparison Results:")
    for opt_name, results in optimizer_results.items():
        print(f"{opt_name:10} - Train Loss: {results['final_loss']:.3f}, "
              f"Test MSE: {results['test_mse']:.3f}, Test R²: {results['test_r2']:.3f}")
    
    print("\nBoston Housing regression example completed!")
    print("Generated files:")
    print("- boston_training_history.png")
    print("- boston_predictions.png")
    print("- boston_residuals_vs_features.png")
    print("- boston_mlp_model.pkl")
    print("- regularization_comparison.png")
    print("- optimizer_comparison_regression.png")


if __name__ == "__main__":
    main() 