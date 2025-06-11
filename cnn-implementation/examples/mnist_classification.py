import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Import our CNN implementation
from core.model import Sequential
from core.layers.base import Conv2D, MaxPool2D, Dense

def load_mnist(num_samples=5000):
    """Load and preprocess MNIST dataset."""
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Take a subset for quick demonstration
    X = X[:num_samples]
    y = y[:num_samples]
    
    # Reshape and normalize images
    X = X.reshape(-1, 1, 28, 28) / 255.0
    
    # Convert labels to one-hot encoding
    y_one_hot = np.zeros((len(y), 10))
    y_one_hot[np.arange(len(y)), y.astype(int)] = 1
    
    return train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

def create_model():
    """Create a simple CNN model for MNIST classification."""
    model = Sequential([
        # First Convolutional Block
        Conv2D(filters=32, kernel_size=3, activation='relu'),
        MaxPool2D(pool_size=2),
        
        # Second Convolutional Block
        Conv2D(filters=64, kernel_size=3, activation='relu'),
        MaxPool2D(pool_size=2),
        
        # Flatten and Dense Layers
        Dense(units=128, activation='relu'),
        Dense(units=10, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """Plot training history."""
    plt.figure(figsize=(10, 4))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_mnist()
    print("Data loaded and preprocessed.")
    
    # Create and compile model
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='cross_entropy'
    )
    print("Model created and compiled.")
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=True
    )
    
    # Evaluate model
    test_loss = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Plot training history
    plot_training_history(history)

if __name__ == '__main__':
    main()