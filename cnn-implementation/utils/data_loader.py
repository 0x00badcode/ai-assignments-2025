"""Data loading utilities for CNN training and evaluation."""

import numpy as np
import os
from typing import Tuple, Optional, List, Dict, Any
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
try:
    from PIL import Image
except ImportError:
    Image = None


class DataLoader:
    """Generic data loader for batch processing."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        """Initialize DataLoader.
        
        Args:
            X: Input data
            y: Target labels
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
        self.n_batches = int(np.ceil(self.n_samples / batch_size))
        self.indices = np.arange(self.n_samples)
        
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        """Make DataLoader iterable."""
        self.current_batch = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        """Get next batch."""
        if self.current_batch >= self.n_batches:
            raise StopIteration
        
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        
        self.current_batch += 1
        return batch_X, batch_y
    
    def __len__(self):
        """Get number of batches."""
        return self.n_batches


def load_mnist(num_samples: Optional[int] = None, 
               normalize: bool = True, 
               one_hot: bool = True,
               test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess MNIST dataset.
    
    Args:
        num_samples: Number of samples to load (None for all)
        normalize: Whether to normalize pixel values
        one_hot: Whether to convert labels to one-hot encoding
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Limit samples if specified
    if num_samples:
        X = X[:num_samples]
        y = y[:num_samples]
    
    # Reshape to image format (N, C, H, W)
    X = X.reshape(-1, 1, 28, 28)
    
    # Normalize pixel values
    if normalize:
        X = X.astype(np.float32) / 255.0
    
    # Convert labels to integers
    y = y.astype(int)
    
    # One-hot encoding
    if one_hot:
        y_encoded = np.zeros((len(y), 10))
        y_encoded[np.arange(len(y)), y] = 1
        y = y_encoded
    
    return train_test_split(X, y, test_size=test_size, random_state=42)


def load_cifar10(data_dir: str = './data', 
                 normalize: bool = True,
                 one_hot: bool = True,
                 test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess CIFAR-10 dataset.
    
    Args:
        data_dir: Directory containing CIFAR-10 data
        normalize: Whether to normalize pixel values
        one_hot: Whether to convert labels to one-hot encoding
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    # Load training batches
    X_train_batches = []
    y_train_batches = []
    
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        if os.path.exists(batch_file):
            batch = unpickle(batch_file)
            X_train_batches.append(batch[b'data'])
            y_train_batches.extend(batch[b'labels'])
    
    if X_train_batches:
        X_train = np.vstack(X_train_batches)
        y_train = np.array(y_train_batches)
    else:
        # Fallback: create dummy CIFAR-10 data for demonstration
        print("CIFAR-10 data not found, creating dummy data for demonstration...")
        X_train = np.random.randint(0, 256, (5000, 3072), dtype=np.uint8)
        y_train = np.random.randint(0, 10, 5000)
    
    # Load test batch
    test_file = os.path.join(data_dir, 'test_batch')
    if os.path.exists(test_file):
        test_batch = unpickle(test_file)
        X_test = test_batch[b'data']
        y_test = np.array(test_batch[b'labels'])
    else:
        # Use part of training data as test
        X_test = X_train[-1000:]
        y_test = y_train[-1000:]
        X_train = X_train[:-1000]
        y_train = y_train[:-1000]
    
    # Reshape to image format (N, C, H, W)
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)
    
    # Normalize pixel values
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
    
    # One-hot encoding
    if one_hot:
        y_train_encoded = np.zeros((len(y_train), 10))
        y_train_encoded[np.arange(len(y_train)), y_train] = 1
        y_train = y_train_encoded
        
        y_test_encoded = np.zeros((len(y_test), 10))
        y_test_encoded[np.arange(len(y_test)), y_test] = 1
        y_test = y_test_encoded
    
    return X_train, X_test, y_train, y_test


def load_lfw_dataset(data_dir: str = './data/lfw',
                     target_size: Tuple[int, int] = (128, 128),
                     normalize: bool = True,
                     test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Labeled Faces in the Wild (LFW) dataset for face recognition.
    
    Args:
        data_dir: Directory containing LFW data
        target_size: Target image size (H, W)
        normalize: Whether to normalize pixel values
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if not Image:
        raise ImportError("PIL is required for image loading")
    
    X = []
    y = []
    label_encoder = LabelEncoder()
    
    if not os.path.exists(data_dir):
        print(f"LFW dataset not found at {data_dir}, creating dummy face data...")
        # Create dummy face data for demonstration
        num_people = 50
        images_per_person = 20
        X = np.random.rand(num_people * images_per_person, 3, *target_size).astype(np.float32)
        y = np.repeat(range(num_people), images_per_person)
        
        if normalize:
            X = X * 255.0 / X.max()
        
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Load real LFW data
    for person_dir in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_dir)
        if os.path.isdir(person_path):
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_path, img_file)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(target_size)
                        img_array = np.array(img).transpose(2, 0, 1)  # HWC to CHW
                        X.append(img_array)
                        y.append(person_dir)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
    
    X = np.array(X, dtype=np.float32)
    y = label_encoder.fit_transform(y)
    
    if normalize:
        X = X / 255.0
    
    return train_test_split(X, y, test_size=test_size, random_state=42)


def create_face_pairs(X: np.ndarray, y: np.ndarray, 
                      num_pairs: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Create pairs of faces for face verification tasks.
    
    Args:
        X: Face images
        y: Face labels
        num_pairs: Number of pairs to create
        
    Returns:
        Tuple of (pairs, labels) where labels indicate if pair is same person
    """
    pairs = []
    labels = []
    
    unique_labels = np.unique(y)
    
    for _ in range(num_pairs):
        if np.random.random() > 0.5:
            # Same person pair
            label = np.random.choice(unique_labels)
            indices = np.where(y == label)[0]
            if len(indices) >= 2:
                idx1, idx2 = np.random.choice(indices, 2, replace=False)
                pairs.append([X[idx1], X[idx2]])
                labels.append(1)
        else:
            # Different person pair
            label1, label2 = np.random.choice(unique_labels, 2, replace=False)
            idx1 = np.random.choice(np.where(y == label1)[0])
            idx2 = np.random.choice(np.where(y == label2)[0])
            pairs.append([X[idx1], X[idx2]])
            labels.append(0)
    
    return np.array(pairs), np.array(labels)


def augment_data(X: np.ndarray, 
                 rotation_range: float = 0.2,
                 width_shift_range: float = 0.1,
                 height_shift_range: float = 0.1,
                 horizontal_flip: bool = False) -> np.ndarray:
    """Apply basic data augmentation to images.
    
    Args:
        X: Input images
        rotation_range: Range of rotation in radians
        width_shift_range: Range of horizontal shift
        height_shift_range: Range of vertical shift
        horizontal_flip: Whether to apply horizontal flip
        
    Returns:
        Augmented images
    """
    augmented = []
    
    for img in X:
        # Original image
        augmented.append(img)
        
        # Add noise
        noise = np.random.normal(0, 0.01, img.shape)
        noisy_img = np.clip(img + noise, 0, 1)
        augmented.append(noisy_img)
        
        # Horizontal flip
        if horizontal_flip and np.random.random() > 0.5:
            flipped = img[:, :, ::-1] if len(img.shape) == 3 else img[:, ::-1]
            augmented.append(flipped)
        
        # Simple brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        bright_img = np.clip(img * brightness_factor, 0, 1)
        augmented.append(bright_img)
    
    return np.array(augmented) 