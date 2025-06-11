"""MobileFaceNet implementation for efficient face recognition."""

import numpy as np
from typing import Optional, Tuple
from .base import BaseModel
from core.layers.base import (Conv2D, DepthwiseConv2D, Dense, BatchNormalization, 
                              Dropout, AvgPool2D, Flatten, Bottleneck)


class MobileFaceNet(BaseModel):
    """MobileFaceNet for efficient face recognition.
    
    Based on: "MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification"
    """
    
    def __init__(self, embedding_size: int = 128, 
                 num_classes: Optional[int] = None,
                 input_size: Tuple[int, int] = (112, 112),
                 dropout_rate: float = 0.2):
        """Initialize MobileFaceNet.
        
        Args:
            embedding_size: Size of the face embedding vector
            num_classes: Number of classes for classification (None for embedding only)
            input_size: Input image size (H, W)
            dropout_rate: Dropout rate
        """
        super().__init__(name="MobileFaceNet")
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        
        self._build_model()
    
    def _build_model(self):
        """Build the MobileFaceNet architecture."""
        # Initial convolution
        self.add(Conv2D(filters=64, kernel_size=3, stride=2, padding='same',
                       activation='relu', kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        
        # Depthwise separable convolution
        self.add(DepthwiseConv2D(kernel_size=3, padding='same',
                               activation='relu', kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        
        # Bottleneck blocks
        # Block 1
        self.add(Bottleneck(in_channels=64, bottleneck_channels=64, out_channels=64,
                          stride=2, activation='relu'))
        
        # Block 2 - 5 (4 repetitions)
        for _ in range(4):
            self.add(Bottleneck(in_channels=64, bottleneck_channels=128, out_channels=128,
                              stride=1, activation='relu'))
        
        # Block 6
        self.add(Bottleneck(in_channels=128, bottleneck_channels=128, out_channels=128,
                          stride=2, activation='relu'))
        
        # Block 7 - 12 (6 repetitions)
        for _ in range(6):
            self.add(Bottleneck(in_channels=128, bottleneck_channels=256, out_channels=256,
                              stride=1, activation='relu'))
        
        # Block 13
        self.add(Bottleneck(in_channels=256, bottleneck_channels=256, out_channels=256,
                          stride=2, activation='relu'))
        
        # Block 14 - 15 (2 repetitions)
        for _ in range(2):
            self.add(Bottleneck(in_channels=256, bottleneck_channels=512, out_channels=512,
                              stride=1, activation='relu'))
        
        # Final convolution
        self.add(Conv2D(filters=512, kernel_size=1, activation='relu',
                       kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        
        # Global average pooling
        self.add(AvgPool2D(pool_size=7))  # Assumes 7x7 feature map
        
        # Flatten
        self.add(Flatten())
        
        # Dropout
        self.add(Dropout(rate=self.dropout_rate))
        
        # Embedding layer
        self.add(Dense(units=self.embedding_size, activation=None,
                      kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        
        # Classification layer (optional)
        if self.num_classes is not None:
            self.add(Dense(units=self.num_classes, activation='softmax',
                          kernel_initializer='he_normal'))
    
    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract face embeddings.
        
        Args:
            x: Input face images
            
        Returns:
            Face embeddings
        """
        # Forward pass up to embedding layer
        output = x
        for layer in self.layers[:-1] if self.num_classes else self.layers:
            output = layer.forward(output)
        
        # Normalize embeddings
        embeddings = output
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray,
                        metric: str = 'cosine') -> float:
        """Compute distance between two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            metric: Distance metric ('cosine', 'euclidean')
            
        Returns:
            Distance between embeddings
        """
        if metric == 'cosine':
            # Cosine similarity -> distance
            similarity = np.dot(embedding1, embedding2)
            return 1.0 - similarity
        elif metric == 'euclidean':
            return np.linalg.norm(embedding1 - embedding2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def verify_faces(self, embedding1: np.ndarray, embedding2: np.ndarray,
                    threshold: float = 0.5, metric: str = 'cosine') -> Tuple[bool, float]:
        """Verify if two face embeddings belong to the same person.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Verification threshold
            metric: Distance metric
            
        Returns:
            Tuple of (is_same_person, distance)
        """
        distance = self.compute_distance(embedding1, embedding2, metric)
        is_same = distance < threshold
        return is_same, distance
    
    def compute_face_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute similarity matrix for a set of face embeddings.
        
        Args:
            embeddings: Face embeddings matrix (N, embedding_size)
            
        Returns:
            Similarity matrix (N, N)
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        
        # Compute cosine similarity matrix
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        return similarity_matrix
    
    @staticmethod
    def triplet_loss(anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray,
                    margin: float = 0.2) -> float:
        """Compute triplet loss for face recognition training.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (same person as anchor)
            negative: Negative embeddings (different person)
            margin: Margin for triplet loss
            
        Returns:
            Triplet loss value
        """
        # Compute distances
        pos_dist = np.linalg.norm(anchor - positive, axis=1)
        neg_dist = np.linalg.norm(anchor - negative, axis=1)
        
        # Triplet loss
        loss = np.maximum(0, pos_dist - neg_dist + margin)
        return np.mean(loss)
    
    @staticmethod
    def center_loss(embeddings: np.ndarray, labels: np.ndarray, 
                   centers: np.ndarray, alpha: float = 0.5) -> Tuple[float, np.ndarray]:
        """Compute center loss for face recognition.
        
        Args:
            embeddings: Face embeddings
            labels: Ground truth labels
            centers: Class centers
            alpha: Learning rate for center updates
            
        Returns:
            Tuple of (center_loss, updated_centers)
        """
        batch_size = embeddings.shape[0]
        
        # Select centers for current batch
        selected_centers = centers[labels]
        
        # Center loss
        center_loss = 0.5 * np.mean(np.sum((embeddings - selected_centers) ** 2, axis=1))
        
        # Update centers
        unique_labels = np.unique(labels)
        updated_centers = centers.copy()
        
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 0:
                diff = np.mean(embeddings[mask] - centers[label], axis=0)
                updated_centers[label] += alpha * diff
        
        return center_loss, updated_centers


class MobileFaceNetV2(MobileFaceNet):
    """Improved version of MobileFaceNet with additional optimizations."""
    
    def __init__(self, embedding_size: int = 128,
                 num_classes: Optional[int] = None,
                 input_size: Tuple[int, int] = (112, 112),
                 dropout_rate: float = 0.2,
                 width_multiplier: float = 1.0):
        """Initialize MobileFaceNetV2.
        
        Args:
            embedding_size: Size of the face embedding vector
            num_classes: Number of classes for classification
            input_size: Input image size (H, W)
            dropout_rate: Dropout rate
            width_multiplier: Width multiplier for channels
        """
        self.width_multiplier = width_multiplier
        super().__init__(embedding_size, num_classes, input_size, dropout_rate)
        self.name = "MobileFaceNetV2"
    
    def _make_divisible(self, v: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
        """Make channel number divisible by divisor."""
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    
    def _build_model(self):
        """Build the improved MobileFaceNetV2 architecture."""
        # Apply width multiplier to channels
        def make_ch(channels):
            return self._make_divisible(int(channels * self.width_multiplier))
        
        # Initial convolution
        self.add(Conv2D(filters=make_ch(64), kernel_size=3, stride=2, padding='same',
                       activation='relu', kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        
        # Inverted residual blocks with different expansion ratios
        # Stage 1
        self.add(Bottleneck(in_channels=make_ch(64), bottleneck_channels=make_ch(64), 
                          out_channels=make_ch(64), stride=1, activation='relu'))
        
        # Stage 2
        self.add(Bottleneck(in_channels=make_ch(64), bottleneck_channels=make_ch(128), 
                          out_channels=make_ch(128), stride=2, activation='relu'))
        for _ in range(3):
            self.add(Bottleneck(in_channels=make_ch(128), bottleneck_channels=make_ch(128), 
                              out_channels=make_ch(128), stride=1, activation='relu'))
        
        # Stage 3
        self.add(Bottleneck(in_channels=make_ch(128), bottleneck_channels=make_ch(256), 
                          out_channels=make_ch(256), stride=2, activation='relu'))
        for _ in range(7):
            self.add(Bottleneck(in_channels=make_ch(256), bottleneck_channels=make_ch(256), 
                              out_channels=make_ch(256), stride=1, activation='relu'))
        
        # Stage 4
        self.add(Bottleneck(in_channels=make_ch(256), bottleneck_channels=make_ch(512), 
                          out_channels=make_ch(512), stride=2, activation='relu'))
        for _ in range(3):
            self.add(Bottleneck(in_channels=make_ch(512), bottleneck_channels=make_ch(512), 
                              out_channels=make_ch(512), stride=1, activation='relu'))
        
        # Final layers
        self.add(Conv2D(filters=make_ch(1024), kernel_size=1, activation='relu',
                       kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        
        # Global average pooling
        self.add(AvgPool2D(pool_size=7))
        self.add(Flatten())
        
        # Embedding with dropout
        self.add(Dropout(rate=self.dropout_rate))
        self.add(Dense(units=self.embedding_size, activation=None,
                      kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        
        # Optional classification head
        if self.num_classes is not None:
            self.add(Dense(units=self.num_classes, activation='softmax',
                          kernel_initializer='he_normal')) 