"""
MNIST Classification Example
Demonstrates the MLP library on the MNIST digit classification task
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add parent directory to path to import mlp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp import MLP, StoppingCriteria
from mlp.optimizers import Adam, SGD, Momentum
from mlp.regularization import L2Regularization, ElasticNetRegularization
from mlp.evaluation import ConfusionMatrix, ClassificationEvaluator


def load_mnist_data():
    """
    Load MNIST data (you'll need to download from Kaggle)
    https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    """
    print("Loading MNIST data...")
    try:
        # Try to load from current directory
        train_data = pd.read_csv('mnist_train.csv')
        test_data = pd.read_csv('mnist_test.csv')
    except FileNotFoundError:
        print("MNIST CSV files not found!")
        print("Please download from: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")
        print("And place mnist_train.csv and mnist_test.csv in the current directory.")
        
        # Generate sample data for demonstration
        print("Generating sample data for demonstration...")
        np.random.seed(42)
        n_samples = 1000
        n_features = 784
        
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 10, n_samples)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Extract features and labels
    X_train = train_data.drop('label', axis=1).values.astype(np.float32) / 255.0  # Normalize
    y_train = train_data['label'].values.astype(np.int32)
    
    X_test = test_data.drop('label', axis=1).values.astype(np.float32) / 255.0  # Normalize
    y_test = test_data['label'].values.astype(np.int32)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    return X_train, X_test, y_train, y_test


def create_one_hot(y, num_classes=10):
    """Convert labels to one-hot encoding"""
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def main():
    print("MNIST Classification with MLP from Scratch")
    print("=" * 50)
    
    # Load data
    X_train, X_test, y_train, y_test = load_mnist_data()
    
    # Convert to one-hot encoding
    y_train_onehot = create_one_hot(y_train)
    y_test_onehot = create_one_hot(y_test)
    
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train_onehot, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train_split.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create and configure the model
    print("\nBuilding MLP model...")
    model = MLP(task_type='classification')
    
    # Add layers
    model.add_first_dense(784, 128, activation='relu', weight_init='he')
    model.add_dropout(0.3)
    model.add_dense(64, activation='relu', weight_init='he')
    model.add_dropout(0.3)
    model.add_dense(10, activation='softmax', weight_init='xavier')
    
    # Compile model
    regularization = L2Regularization(lambda_reg=0.001)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        regularization=regularization,
        learning_rate=0.001
    )
    
    # Print model summary
    model.summary()
    
    # Set up early stopping
    early_stopping = StoppingCriteria(
        patience=10,
        min_delta=0.001,
        monitor='val_loss',
        mode='min'
    )
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train_split, y_train_split,
        epochs=50,
        batch_size=128,
        validation_data=(X_val, y_val),
        verbose=1,
        early_stopping=early_stopping,
        shuffle=True
    )
    
    # Plot training history
    model.plot_history(save_path='mnist_training_history.png')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = model.evaluate(X_test, y_test_onehot)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Get predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Create confusion matrix
    print("\nGenerating confusion matrix...")
    class_names = [str(i) for i in range(10)]
    cm = ConfusionMatrix(y_test, predicted_classes, class_names)
    
    print(cm.classification_report())
    
    # Plot confusion matrix
    cm.plot(normalize=True, save_path='mnist_confusion_matrix.png')
    
    # Comprehensive evaluation
    evaluator = ClassificationEvaluator(
        y_test, predicted_classes, predictions, class_names
    )
    
    metrics = evaluator.get_metrics_dict()
    print("\nDetailed Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save the model
    print("\nSaving model...")
    model.save('mnist_mlp_model.pkl')
    
    # Demonstrate different optimizers
    print("\nComparing different optimizers...")
    optimizers = [
        ('SGD', SGD(learning_rate=0.01)),
        ('Momentum', Momentum(learning_rate=0.01, momentum=0.9)),
        ('Adam', Adam(learning_rate=0.001))
    ]
    
    optimizer_results = {}
    
    for opt_name, optimizer in optimizers:
        print(f"\nTraining with {opt_name}...")
        
        # Create new model
        model_opt = MLP(task_type='classification')
        model_opt.add_first_dense(784, 64, activation='relu')
        model_opt.add_dense(32, activation='relu')
        model_opt.add_dense(10, activation='softmax')
        
        model_opt.compile(optimizer=optimizer, loss='categorical_crossentropy')
        
        # Train for fewer epochs for comparison
        history_opt = model_opt.fit(
            X_train_split[:5000], y_train_split[:5000],  # Use subset for faster training
            epochs=10,
            batch_size=64,
            validation_data=(X_val[:1000], y_val[:1000]),
            verbose=0
        )
        
        # Evaluate
        test_metrics_opt = model_opt.evaluate(X_test[:1000], y_test_onehot[:1000])
        optimizer_results[opt_name] = {
            'final_loss': history_opt['loss'][-1],
            'final_accuracy': history_opt['accuracy'][-1],
            'test_accuracy': test_metrics_opt['accuracy']
        }
    
    # Plot optimizer comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    opt_names = list(optimizer_results.keys())
    train_accuracies = [optimizer_results[name]['final_accuracy'] for name in opt_names]
    test_accuracies = [optimizer_results[name]['test_accuracy'] for name in opt_names]
    
    axes[0].bar(opt_names, train_accuracies)
    axes[0].set_title('Training Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    
    axes[1].bar(opt_names, test_accuracies)
    axes[1].set_title('Test Accuracy Comparison')
    axes[1].set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nOptimizer Comparison Results:")
    for opt_name, results in optimizer_results.items():
        print(f"{opt_name:10} - Train Acc: {results['final_accuracy']:.3f}, "
              f"Test Acc: {results['test_accuracy']:.3f}")
    
    print("\nMNIST classification example completed!")
    print("Generated files:")
    print("- mnist_training_history.png")
    print("- mnist_confusion_matrix.png")
    print("- mnist_mlp_model.pkl")
    print("- optimizer_comparison.png")


if __name__ == "__main__":
    main() 