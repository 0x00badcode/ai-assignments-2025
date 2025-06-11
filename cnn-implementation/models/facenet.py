"""FaceNet implementation for face recognition and verification."""

import numpy as np
from typing import Optional, Tuple, List
from .base import BaseModel
from core.layers.base import (Conv2D, MaxPool2D, AvgPool2D, Dense, BatchNormalization, 
                              Dropout, Flatten, InceptionModule, ResidualBlock)


class FaceNet(BaseModel):
    """FaceNet implementation for face recognition.
    
    Based on: "FaceNet: A Unified Embedding for Face Recognition and Clustering"
    Architecture inspired by Inception-ResNet
    """
    
    def __init__(self, embedding_size: int = 128,
                 input_size: Tuple[int, int] = (160, 160),
                 architecture: str = 'inception_resnet_v1'):
        """Initialize FaceNet.
        
        Args:
            embedding_size: Size of the face embedding vector
            input_size: Input image size (H, W)
            architecture: Architecture type ('inception_resnet_v1' or 'inception_v4')
        """
        super().__init__(name="FaceNet")
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.architecture = architecture
        
        self._build_model()
    
    def _build_model(self):
        """Build the FaceNet architecture."""
        if self.architecture == 'inception_resnet_v1':
            self._build_inception_resnet_v1()
        else:
            self._build_inception_v4()
    
    def _build_inception_resnet_v1(self):
        """Build Inception-ResNet-v1 architecture for FaceNet."""
        # Stem layers
        self.add(Conv2D(32, 3, stride=2, padding='valid', activation='relu'))
        self.add(Conv2D(32, 3, padding='valid', activation='relu'))
        self.add(Conv2D(64, 3, padding='same', activation='relu'))
        self.add(MaxPool2D(3, stride=2, padding='valid'))
        
        self.add(Conv2D(80, 1, padding='valid', activation='relu'))
        self.add(Conv2D(192, 3, padding='valid', activation='relu'))
        self.add(Conv2D(256, 3, stride=2, padding='valid', activation='relu'))
        
        # Inception-ResNet-A blocks (5x)
        for _ in range(5):
            self.add(InceptionResNetA())
        
        # Reduction-A
        self.add(ReductionA())
        
        # Inception-ResNet-B blocks (10x)
        for _ in range(10):
            self.add(InceptionResNetB())
        
        # Reduction-B
        self.add(ReductionB())
        
        # Inception-ResNet-C blocks (5x)
        for _ in range(5):
            self.add(InceptionResNetC())
        
        # Final layers
        self.add(AvgPool2D(8))  # Global average pooling
        self.add(Dropout(0.6))
        self.add(Flatten())
        self.add(Dense(self.embedding_size, activation=None))
        self.add(BatchNormalization())
    
    def _build_inception_v4(self):
        """Build Inception-v4 architecture for FaceNet."""
        # Stem
        self.add(Conv2D(32, 3, stride=2, padding='valid', activation='relu'))
        self.add(Conv2D(32, 3, padding='valid', activation='relu'))
        self.add(Conv2D(64, 3, padding='same', activation='relu'))
        
        # Mixed layers (simplified Inception v4)
        self.add(InceptionModule(64, 48, 64, 16, 32, 32, activation='relu'))
        self.add(InceptionModule(64, 48, 64, 16, 32, 32, activation='relu'))
        self.add(InceptionModule(64, 48, 64, 16, 32, 32, activation='relu'))
        
        self.add(MaxPool2D(3, stride=2))
        
        self.add(InceptionModule(128, 96, 128, 32, 64, 64, activation='relu'))
        self.add(InceptionModule(128, 96, 128, 32, 64, 64, activation='relu'))
        
        self.add(MaxPool2D(3, stride=2))
        
        self.add(InceptionModule(256, 192, 256, 64, 128, 128, activation='relu'))
        self.add(InceptionModule(256, 192, 256, 64, 128, 128, activation='relu'))
        
        # Final layers
        self.add(AvgPool2D(8))
        self.add(Dropout(0.8))
        self.add(Flatten())
        self.add(Dense(self.embedding_size, activation=None))
        self.add(BatchNormalization())
    
    def extract_embeddings(self, x: np.ndarray) -> np.ndarray:
        """Extract normalized face embeddings.
        
        Args:
            x: Input face images
            
        Returns:
            L2-normalized face embeddings
        """
        # Forward pass
        embeddings = self.forward(x, training=False)
        
        # L2 normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        
        return normalized_embeddings
    
    def compute_triplet_loss(self, anchor: np.ndarray, positive: np.ndarray, 
                           negative: np.ndarray, margin: float = 0.2) -> float:
        """Compute triplet loss for FaceNet training.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (same person)
            negative: Negative embeddings (different person)
            margin: Margin for triplet loss
            
        Returns:
            Triplet loss value
        """
        # Extract embeddings
        anchor_emb = self.extract_embeddings(anchor)
        positive_emb = self.extract_embeddings(positive)
        negative_emb = self.extract_embeddings(negative)
        
        # Compute squared L2 distances
        pos_dist = np.sum((anchor_emb - positive_emb) ** 2, axis=1)
        neg_dist = np.sum((anchor_emb - negative_emb) ** 2, axis=1)
        
        # Triplet loss
        loss = np.maximum(0, pos_dist - neg_dist + margin)
        return np.mean(loss)
    
    def generate_triplets(self, images: np.ndarray, labels: np.ndarray, 
                         batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate triplets for training.
        
        Args:
            images: Face images
            labels: Identity labels
            batch_size: Number of triplets to generate
            
        Returns:
            Tuple of (anchor, positive, negative) triplets
        """
        anchors, positives, negatives = [], [], []
        unique_labels = np.unique(labels)
        
        for _ in range(batch_size):
            # Select anchor
            anchor_label = np.random.choice(unique_labels)
            anchor_indices = np.where(labels == anchor_label)[0]
            
            if len(anchor_indices) < 2:
                continue
                
            anchor_idx, positive_idx = np.random.choice(anchor_indices, 2, replace=False)
            
            # Select negative
            negative_label = np.random.choice(unique_labels[unique_labels != anchor_label])
            negative_indices = np.where(labels == negative_label)[0]
            negative_idx = np.random.choice(negative_indices)
            
            anchors.append(images[anchor_idx])
            positives.append(images[positive_idx])
            negatives.append(images[negative_idx])
        
        return np.array(anchors), np.array(positives), np.array(negatives)
    
    def verify_faces(self, face1: np.ndarray, face2: np.ndarray, 
                    threshold: float = 1.0) -> Tuple[bool, float]:
        """Verify if two faces belong to the same person.
        
        Args:
            face1: First face image
            face2: Second face image
            threshold: Distance threshold for verification
            
        Returns:
            Tuple of (is_same_person, distance)
        """
        # Extract embeddings
        emb1 = self.extract_embeddings(face1.reshape(1, *face1.shape))
        emb2 = self.extract_embeddings(face2.reshape(1, *face2.shape))
        
        # Compute L2 distance
        distance = np.linalg.norm(emb1 - emb2)
        
        # Verify if same person
        is_same = distance < threshold
        
        return is_same, distance


class InceptionResNetA(BaseModel):
    """Inception-ResNet-A block."""
    
    def __init__(self):
        super().__init__(name="InceptionResNetA")
        self._build_block()
    
    def _build_block(self):
        # Branch 1
        self.conv1_1 = Conv2D(32, 1, activation='relu')
        
        # Branch 2
        self.conv2_1 = Conv2D(32, 1, activation='relu')
        self.conv2_2 = Conv2D(32, 3, padding='same', activation='relu')
        
        # Branch 3
        self.conv3_1 = Conv2D(32, 1, activation='relu')
        self.conv3_2 = Conv2D(32, 3, padding='same', activation='relu')
        self.conv3_3 = Conv2D(32, 3, padding='same', activation='relu')
        
        # Linear projection
        self.conv_linear = Conv2D(256, 1, activation=None)
        self.batch_norm = BatchNormalization()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        # Branch 1
        branch1 = self.conv1_1.forward(x)
        
        # Branch 2
        branch2 = self.conv2_1.forward(x)
        branch2 = self.conv2_2.forward(branch2)
        
        # Branch 3
        branch3 = self.conv3_1.forward(x)
        branch3 = self.conv3_2.forward(branch3)
        branch3 = self.conv3_3.forward(branch3)
        
        # Concatenate branches
        concat = np.concatenate([branch1, branch2, branch3], axis=1)
        
        # Linear projection
        projected = self.conv_linear.forward(concat)
        projected = self.batch_norm.forward(projected)
        
        # Residual connection
        output = x + projected
        
        # ReLU activation
        output = np.maximum(0, output)
        
        self.output = output
        return output


class InceptionResNetB(BaseModel):
    """Inception-ResNet-B block."""
    
    def __init__(self):
        super().__init__(name="InceptionResNetB")
        self._build_block()
    
    def _build_block(self):
        # Branch 1
        self.conv1_1 = Conv2D(128, 1, activation='relu')
        
        # Branch 2
        self.conv2_1 = Conv2D(128, 1, activation='relu')
        self.conv2_2 = Conv2D(128, (1, 7), padding='same', activation='relu')
        self.conv2_3 = Conv2D(128, (7, 1), padding='same', activation='relu')
        
        # Linear projection
        self.conv_linear = Conv2D(896, 1, activation=None)
        self.batch_norm = BatchNormalization()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        # Branch 1
        branch1 = self.conv1_1.forward(x)
        
        # Branch 2
        branch2 = self.conv2_1.forward(x)
        branch2 = self.conv2_2.forward(branch2)
        branch2 = self.conv2_3.forward(branch2)
        
        # Concatenate branches
        concat = np.concatenate([branch1, branch2], axis=1)
        
        # Linear projection
        projected = self.conv_linear.forward(concat)
        projected = self.batch_norm.forward(projected)
        
        # Residual connection
        output = x + projected
        
        # ReLU activation
        output = np.maximum(0, output)
        
        self.output = output
        return output


class InceptionResNetC(BaseModel):
    """Inception-ResNet-C block."""
    
    def __init__(self):
        super().__init__(name="InceptionResNetC")
        self._build_block()
    
    def _build_block(self):
        # Branch 1
        self.conv1_1 = Conv2D(192, 1, activation='relu')
        
        # Branch 2
        self.conv2_1 = Conv2D(192, 1, activation='relu')
        self.conv2_2 = Conv2D(192, (1, 3), padding='same', activation='relu')
        self.conv2_3 = Conv2D(192, (3, 1), padding='same', activation='relu')
        
        # Linear projection
        self.conv_linear = Conv2D(1792, 1, activation=None)
        self.batch_norm = BatchNormalization()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        # Branch 1
        branch1 = self.conv1_1.forward(x)
        
        # Branch 2
        branch2 = self.conv2_1.forward(x)
        branch2 = self.conv2_2.forward(branch2)
        branch2 = self.conv2_3.forward(branch2)
        
        # Concatenate branches
        concat = np.concatenate([branch1, branch2], axis=1)
        
        # Linear projection
        projected = self.conv_linear.forward(concat)
        projected = self.batch_norm.forward(projected)
        
        # Residual connection
        output = x + projected
        
        # ReLU activation
        output = np.maximum(0, output)
        
        self.output = output
        return output


class ReductionA(BaseModel):
    """Reduction-A block for downsampling."""
    
    def __init__(self):
        super().__init__(name="ReductionA")
        self._build_block()
    
    def _build_block(self):
        # Branch 1
        self.conv1 = Conv2D(384, 3, stride=2, padding='valid', activation='relu')
        
        # Branch 2
        self.conv2_1 = Conv2D(192, 1, activation='relu')
        self.conv2_2 = Conv2D(192, 3, padding='same', activation='relu')
        self.conv2_3 = Conv2D(256, 3, stride=2, padding='valid', activation='relu')
        
        # Branch 3: MaxPool
        self.maxpool = MaxPool2D(3, stride=2, padding='valid')
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        # Branch 1
        branch1 = self.conv1.forward(x)
        
        # Branch 2
        branch2 = self.conv2_1.forward(x)
        branch2 = self.conv2_2.forward(branch2)
        branch2 = self.conv2_3.forward(branch2)
        
        # Branch 3
        branch3 = self.maxpool.forward(x)
        
        # Concatenate branches
        output = np.concatenate([branch1, branch2, branch3], axis=1)
        
        self.output = output
        return output


class ReductionB(BaseModel):
    """Reduction-B block for downsampling."""
    
    def __init__(self):
        super().__init__(name="ReductionB")
        self._build_block()
    
    def _build_block(self):
        # Branch 1
        self.conv1_1 = Conv2D(256, 1, activation='relu')
        self.conv1_2 = Conv2D(384, 3, stride=2, padding='valid', activation='relu')
        
        # Branch 2
        self.conv2_1 = Conv2D(256, 1, activation='relu')
        self.conv2_2 = Conv2D(256, 3, stride=2, padding='valid', activation='relu')
        
        # Branch 3
        self.conv3_1 = Conv2D(256, 1, activation='relu')
        self.conv3_2 = Conv2D(256, 3, padding='same', activation='relu')
        self.conv3_3 = Conv2D(256, 3, stride=2, padding='valid', activation='relu')
        
        # Branch 4: MaxPool
        self.maxpool = MaxPool2D(3, stride=2, padding='valid')
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        # Branch 1
        branch1 = self.conv1_1.forward(x)
        branch1 = self.conv1_2.forward(branch1)
        
        # Branch 2
        branch2 = self.conv2_1.forward(x)
        branch2 = self.conv2_2.forward(branch2)
        
        # Branch 3
        branch3 = self.conv3_1.forward(x)
        branch3 = self.conv3_2.forward(branch3)
        branch3 = self.conv3_3.forward(branch3)
        
        # Branch 4
        branch4 = self.maxpool.forward(x)
        
        # Concatenate branches
        output = np.concatenate([branch1, branch2, branch3, branch4], axis=1)
        
        self.output = output
        return output 