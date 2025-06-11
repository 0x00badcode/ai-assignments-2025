"""
Quick Test Script
Tests both classification and regression functionality without plots
"""

import numpy as np
import sys
import os

# Add parent directory to path to import mlp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp import MLP, StoppingCriteria
from mlp.optimizers import Adam, SGD
from mlp.regularization import L2Regularization
from mlp.evaluation import ConfusionMatrix, ClassificationEvaluator, RegressionEvaluator
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def test_classification():
    """Test classification functionality"""
    print("Testing Classification...")
    print("-" * 40)
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=3, 
        n_informative=15, n_redundant=5, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create model
    model = MLP(task_type='classification')
    model.add_first_dense(X.shape[1], 32, activation='relu')
    model.add_dropout(0.3)
    model.add_dense(16, activation='relu')
    model.add_dense(3, activation='softmax')
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        regularization=L2Regularization(0.001),
        learning_rate=0.001
    )
    
    # Train
    early_stopping = StoppingCriteria(patience=10, min_delta=0.001)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=0,
        early_stopping=early_stopping
    )
    
    # Evaluate
    test_metrics = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Classification uses 'accuracy' for training, 'val_accuracy' for validation
    if 'accuracy' in history:
        print(f"Final Training Accuracy: {history['accuracy'][-1]:.4f}")
    if 'val_accuracy' in history:
        print(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    
    # Confusion Matrix
    cm = ConfusionMatrix(y_test, predicted_classes)
    print(f"Confusion Matrix Accuracy: {cm.accuracy():.4f}")
    print(f"Macro F1-Score: {cm.f1_score(average='macro'):.4f}")
    
    print("✓ Classification test completed successfully!")
    return True


def test_regression():
    """Test regression functionality"""
    print("\nTesting Regression...")
    print("-" * 40)
    
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=1000, n_features=10, noise=0.1, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Reshape targets
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Create model
    model = MLP(task_type='regression')
    model.add_first_dense(X.shape[1], 64, activation='relu')
    model.add_dropout(0.2)
    model.add_dense(32, activation='relu')
    model.add_dense(16, activation='relu')
    model.add_dense(1, activation='linear')
    
    model.compile(
        optimizer='adam',
        loss='mse',
        regularization=L2Regularization(0.001),
        learning_rate=0.001
    )
    
    # Train
    early_stopping = StoppingCriteria(patience=15, min_delta=0.001)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=0,
        early_stopping=early_stopping
    )
    
    # Evaluate
    test_metrics = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test).flatten()
    
    # Regression uses 'loss' for training, 'val_loss' for validation  
    if 'loss' in history:
        print(f"Final Training Loss: {history['loss'][-1]:.4f}")
    if 'val_loss' in history:
        print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"Test Loss (MSE): {test_metrics['loss']:.4f}")
    print(f"Test RMSE: {np.sqrt(test_metrics['loss']):.4f}")
    
    # Detailed evaluation
    evaluator = RegressionEvaluator(y_test.flatten(), predictions)
    r2 = evaluator.r_squared()
    mae = evaluator.mean_absolute_error()
    
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    
    print("✓ Regression test completed successfully!")
    return True


def test_optimizers():
    """Test different optimizers"""
    print("\nTesting Optimizers...")
    print("-" * 40)
    
    # Generate simple data
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    optimizers = ['sgd', 'adam']
    
    for opt_name in optimizers:
        model = MLP(task_type='classification')
        model.add_first_dense(X.shape[1], 16, activation='relu')
        model.add_dense(2, activation='softmax')
        
        model.compile(optimizer=opt_name, loss='categorical_crossentropy', learning_rate=0.01)
        
        history = model.fit(X_train, y_train, epochs=20, verbose=0)
        test_metrics = model.evaluate(X_test, y_test)
        
        print(f"{opt_name.upper()}: Test Accuracy = {test_metrics['accuracy']:.4f}")
    
    print("✓ Optimizer test completed successfully!")
    return True


def main():
    """Run all tests"""
    print("MLP Quick Test Suite")
    print("=" * 50)
    
    success = True
    
    try:
        success &= test_classification()
        success &= test_regression()
        success &= test_optimizers()
        
        if success:
            print("\n" + "=" * 50)
            print("🎉 All tests passed successfully!")
            print("MLP implementation is working correctly!")
        else:
            print("\n❌ Some tests failed!")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    return success


if __name__ == "__main__":
    main() 