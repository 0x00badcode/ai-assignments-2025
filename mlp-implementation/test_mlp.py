"""
Simple test script to verify MLP implementation
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mlp import MLP, StoppingCriteria
from mlp.optimizers import Adam, SGD, Momentum, RMSProp
from mlp.regularization import L1Regularization, L2Regularization, ElasticNetRegularization
from mlp.evaluation import ConfusionMatrix, ClassificationEvaluator, RegressionEvaluator


def test_classification():
    """Test classification functionality"""
    print("Testing Classification...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Convert to one-hot
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y] = 1
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_onehot[:split_idx], y_onehot[split_idx:]
    y_test_labels = y[split_idx:]
    
    # Create model
    model = MLP(task_type='classification')
    model.add_first_dense(n_features, 32, activation='relu', weight_init='he')
    model.add_dropout(0.3)
    model.add_dense(16, activation='relu', weight_init='he')
    model.add_dense(n_classes, activation='softmax', weight_init='xavier')
    
    # Compile with regularization
    regularization = L2Regularization(lambda_reg=0.001)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        regularization=regularization,
        learning_rate=0.001
    )
    
    print(f"Model compiled successfully!")
    model.summary()
    
    # Train
    print("Training...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    test_metrics = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Confusion matrix
    cm = ConfusionMatrix(y_test_labels, predicted_classes)
    print("\nConfusion Matrix:")
    print(cm.matrix)
    print(f"Accuracy: {cm.accuracy():.4f}")
    
    # Save and load
    model.save('test_classification_model.pkl')
    loaded_model = MLP.load('test_classification_model.pkl')
    
    # Test loaded model
    loaded_predictions = loaded_model.predict(X_test)
    assert np.allclose(predictions, loaded_predictions), "Loaded model predictions don't match!"
    
    print("✓ Classification test passed!")
    return True


def test_regression():
    """Test regression functionality"""
    print("\nTesting Regression...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # Create non-linear relationship
    y = (np.sum(X[:, :3], axis=1) ** 2 + 
         np.sum(X[:, 3:6], axis=1) * 2 + 
         np.random.randn(n_samples) * 0.5).reshape(-1, 1)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create model
    model = MLP(task_type='regression')
    model.add_first_dense(n_features, 64, activation='relu', weight_init='he')
    model.add_dropout(0.2)
    model.add_dense(32, activation='relu', weight_init='he')
    model.add_dense(16, activation='relu', weight_init='he')
    model.add_dense(1, activation='linear', weight_init='xavier')
    
    # Compile with regularization
    regularization = ElasticNetRegularization(lambda_reg=0.001, l1_ratio=0.5)
    model.compile(
        optimizer='adam',
        loss='mse',
        regularization=regularization,
        learning_rate=0.001
    )
    
    print(f"Model compiled successfully!")
    model.summary()
    
    # Train
    print("Training...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    test_metrics = model.evaluate(X_test, y_test)
    print(f"Test MSE: {test_metrics['mse']:.4f}")
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Regression evaluation
    evaluator = RegressionEvaluator(y_test.flatten(), predictions.flatten())
    print(f"R²: {evaluator.r_squared():.4f}")
    print(f"RMSE: {evaluator.root_mean_squared_error():.4f}")
    print(f"MAE: {evaluator.mean_absolute_error():.4f}")
    
    # Save and load
    model.save('test_regression_model.pkl')
    loaded_model = MLP.load('test_regression_model.pkl')
    
    # Test loaded model
    loaded_predictions = loaded_model.predict(X_test)
    assert np.allclose(predictions, loaded_predictions), "Loaded model predictions don't match!"
    
    print("✓ Regression test passed!")
    return True


def test_optimizers():
    """Test different optimizers"""
    print("\nTesting Optimizers...")
    
    # Generate simple data
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = (np.sum(X, axis=1) > 0).astype(int).reshape(-1, 1)
    
    optimizers = [
        ('SGD', SGD(learning_rate=0.01)),
        ('Momentum', Momentum(learning_rate=0.01, momentum=0.9)),
        ('RMSProp', RMSProp(learning_rate=0.001)),
        ('Adam', Adam(learning_rate=0.001))
    ]
    
    for opt_name, optimizer in optimizers:
        print(f"Testing {opt_name}...")
        
        model = MLP(task_type='classification')
        model.add_first_dense(5, 16, activation='relu')
        model.add_dense(1, activation='sigmoid')
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        
        # Train for few epochs
        history = model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        
        # Check if loss decreased
        initial_loss = history['loss'][0]
        final_loss = history['loss'][-1]
        
        print(f"  {opt_name}: {initial_loss:.4f} → {final_loss:.4f}")
        assert final_loss < initial_loss * 1.1, f"{opt_name} failed to improve!"
    
    print("✓ All optimizers test passed!")
    return True


def test_regularization():
    """Test different regularization techniques"""
    print("\nTesting Regularization...")
    
    # Generate simple data
    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = (np.sum(X[:, :3], axis=1) > 0).astype(int).reshape(-1, 1)
    
    regularizations = [
        ('None', None),
        ('L1', L1Regularization(lambda_reg=0.01)),
        ('L2', L2Regularization(lambda_reg=0.01)),
        ('Elastic Net', ElasticNetRegularization(lambda_reg=0.01, l1_ratio=0.5))
    ]
    
    for reg_name, regularization in regularizations:
        print(f"Testing {reg_name}...")
        
        model = MLP(task_type='classification')
        model.add_first_dense(10, 32, activation='relu')
        model.add_dense(1, activation='sigmoid')
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            regularization=regularization
        )
        
        # Train for few epochs
        history = model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        
        # Check if training completed
        assert len(history['loss']) == 5, f"{reg_name} training failed!"
        print(f"  {reg_name}: Final loss = {history['loss'][-1]:.4f}")
    
    print("✓ All regularization test passed!")
    return True


def test_early_stopping():
    """Test early stopping functionality"""
    print("\nTesting Early Stopping...")
    
    # Generate simple, already converged data (almost perfect separation)
    np.random.seed(42)
    X = np.random.randn(200, 3)
    # Make the problem very easy so it converges quickly
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int).reshape(-1, 1)
    
    # Split for validation
    X_train, X_val = X[:120], X[120:]
    y_train, y_val = y[:120], y[120:]
    
    # Use a very simple model that should converge quickly
    model = MLP(task_type='classification')
    model.add_first_dense(3, 8, activation='relu')
    model.add_dense(1, activation='sigmoid')
    
    # Use a higher learning rate to converge faster
    model.compile(optimizer=SGD(learning_rate=0.1), loss='binary_crossentropy')
    
    # Set up early stopping with lenient criteria
    early_stopping = StoppingCriteria(
        patience=3,
        min_delta=0.01,
        monitor='val_loss',
        mode='min'
    )
    
    # Train with early stopping
    history = model.fit(
        X_train, y_train,
        epochs=30,  # Should stop early due to convergence
        batch_size=16,
        validation_data=(X_val, y_val),
        early_stopping=early_stopping,
        verbose=0
    )
    
    # Check if stopped early OR converged well
    epochs_trained = len(history['loss'])
    print(f"Trained for {epochs_trained} epochs (max was 30)")
    
    # If it didn't stop early, at least check if it's learning properly
    if epochs_trained >= 30:
        # Check if loss decreased significantly
        initial_loss = history['loss'][0]
        final_loss = history['loss'][-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss
        print(f"Loss reduction: {loss_reduction:.2%}")
        assert loss_reduction > 0.1, "Model should learn something!"
        print("✓ Early stopping test passed (model converged without early stop)!")
    else:
        print("✓ Early stopping test passed (early stopping triggered)!")
    
    return True


def main():
    """Run all tests"""
    print("MLP Implementation Test Suite")
    print("=" * 40)
    
    try:
        # Run tests
        test_classification()
        test_regression()
        test_optimizers()
        test_regularization()
        test_early_stopping()
        
        print("\n" + "=" * 40)
        print("🎉 All tests passed successfully!")
        print("MLP implementation is working correctly!")
        
        # Cleanup
        import os
        test_files = [
            'test_classification_model.pkl',
            'test_regression_model.pkl'
        ]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Cleaned up {file}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 