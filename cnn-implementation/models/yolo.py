"""YOLO V4/V5 implementation for object detection."""

import numpy as np
from typing import Optional, Tuple, List, Dict
from .base import BaseModel
from core.layers.base import (Conv2D, MaxPool2D, AvgPool2D, Dense, BatchNormalization, 
                              Dropout, Flatten, ResidualBlock, DepthwiseConv2D)


class YOLOv4(BaseModel):
    """YOLOv4 implementation for object detection.
    
    Based on: "YOLOv4: Optimal Speed and Accuracy of Object Detection"
    """
    
    def __init__(self, num_classes: int = 80, 
                 input_size: Tuple[int, int] = (608, 608),
                 anchors: Optional[List[List[int]]] = None):
        """Initialize YOLOv4.
        
        Args:
            num_classes: Number of object classes
            input_size: Input image size (H, W)
            anchors: Anchor boxes for different scales
        """
        super().__init__(name="YOLOv4")
        self.num_classes = num_classes
        self.input_size = input_size
        self.anchors = anchors or self._default_anchors()
        self.num_anchors = len(self.anchors[0])
        
        self._build_model()
    
    def _default_anchors(self) -> List[List[int]]:
        """Default anchor boxes for COCO dataset."""
        return [
            [(12, 16), (19, 36), (40, 28)],      # Small objects
            [(36, 75), (76, 55), (72, 146)],    # Medium objects  
            [(142, 110), (192, 243), (459, 401)] # Large objects
        ]
    
    def _build_model(self):
        """Build YOLOv4 architecture with CSPDarknet53 backbone."""
        # CSPDarknet53 Backbone
        self._build_backbone()
        
        # Neck (PANet)
        self._build_neck()
        
        # Head (Detection layers)
        self._build_head()
    
    def _build_backbone(self):
        """Build CSPDarknet53 backbone."""
        # Stem
        self.add(Conv2D(32, 3, padding='same', activation='relu'))
        self.add(BatchNormalization())
        
        # Stage 1
        self.add(Conv2D(64, 3, stride=2, padding='same', activation='relu'))
        self.add(BatchNormalization())
        self.add(CSPBlock(64, 64, n_blocks=1))
        
        # Stage 2
        self.add(Conv2D(128, 3, stride=2, padding='same', activation='relu'))
        self.add(BatchNormalization())
        self.add(CSPBlock(128, 128, n_blocks=2))
        
        # Stage 3
        self.add(Conv2D(256, 3, stride=2, padding='same', activation='relu'))
        self.add(BatchNormalization())
        self.add(CSPBlock(256, 256, n_blocks=8))
        
        # Stage 4
        self.add(Conv2D(512, 3, stride=2, padding='same', activation='relu'))
        self.add(BatchNormalization())
        self.add(CSPBlock(512, 512, n_blocks=8))
        
        # Stage 5
        self.add(Conv2D(1024, 3, stride=2, padding='same', activation='relu'))
        self.add(BatchNormalization())
        self.add(CSPBlock(1024, 1024, n_blocks=4))
    
    def _build_neck(self):
        """Build PANet neck for feature pyramid."""
        # SPP (Spatial Pyramid Pooling)
        self.add(SPPBlock(1024))
        
        # PANet layers
        self.add(Conv2D(512, 1, activation='relu'))
        self.add(BatchNormalization())
        self.add(Conv2D(1024, 3, padding='same', activation='relu'))
        self.add(BatchNormalization())
        self.add(Conv2D(512, 1, activation='relu'))
        self.add(BatchNormalization())
    
    def _build_head(self):
        """Build detection head."""
        # Three detection layers for different scales
        # Large objects (13x13)
        self.add(DetectionLayer(self.num_classes, self.num_anchors, 512))
        
        # Medium objects (26x26) 
        self.add(Conv2D(256, 1, activation='relu'))
        self.add(BatchNormalization())
        # Upsample + concatenate (simplified)
        self.add(DetectionLayer(self.num_classes, self.num_anchors, 256))
        
        # Small objects (52x52)
        self.add(Conv2D(128, 1, activation='relu'))
        self.add(BatchNormalization())
        # Upsample + concatenate (simplified)
        self.add(DetectionLayer(self.num_classes, self.num_anchors, 128))
    
    def predict_boxes(self, x: np.ndarray, 
                     conf_threshold: float = 0.5,
                     nms_threshold: float = 0.4) -> List[Dict]:
        """Predict bounding boxes for objects.
        
        Args:
            x: Input images
            conf_threshold: Confidence threshold
            nms_threshold: NMS threshold
            
        Returns:
            List of detected objects per image
        """
        # Forward pass
        outputs = self.forward(x, training=False)
        
        # Post-process outputs
        detections = []
        batch_size = x.shape[0]
        
        for i in range(batch_size):
            image_detections = self._post_process_output(
                outputs[i], conf_threshold, nms_threshold
            )
            detections.append(image_detections)
        
        return detections
    
    def _post_process_output(self, output: np.ndarray,
                           conf_threshold: float,
                           nms_threshold: float) -> List[Dict]:
        """Post-process YOLO output to get bounding boxes."""
        # Parse YOLO output
        boxes, confidences, class_ids = self._parse_yolo_output(output)
        
        # Filter by confidence
        valid_indices = confidences > conf_threshold
        boxes = boxes[valid_indices]
        confidences = confidences[valid_indices]
        class_ids = class_ids[valid_indices]
        
        # Apply Non-Maximum Suppression
        keep_indices = self._nms(boxes, confidences, nms_threshold)
        
        # Format final detections
        detections = []
        for idx in keep_indices:
            detection = {
                'bbox': boxes[idx],
                'confidence': confidences[idx],
                'class_id': class_ids[idx],
                'class_name': f'class_{class_ids[idx]}'
            }
            detections.append(detection)
        
        return detections
    
    def _parse_yolo_output(self, output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse YOLO output tensor to boxes, confidences, and class IDs."""
        # Simplified parsing - assumes output is already reshaped
        num_predictions = output.shape[0]
        
        # Extract box coordinates, confidence, and class probabilities
        boxes = output[:, :4]  # x, y, w, h
        confidences = output[:, 4]  # objectness score
        class_probs = output[:, 5:]  # class probabilities
        
        # Get class IDs and scores
        class_ids = np.argmax(class_probs, axis=1)
        class_scores = np.max(class_probs, axis=1)
        
        # Final confidence = objectness * class_score
        final_confidences = confidences * class_scores
        
        return boxes, final_confidences, class_ids
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, 
            threshold: float) -> List[int]:
        """Non-Maximum Suppression."""
        if len(boxes) == 0:
            return []
        
        # Convert to (x1, y1, x2, y2) format
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        areas = (x2 - x1) * (y2 - y1)
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[current], x1[indices[1:]])
            yy1 = np.maximum(y1[current], y1[indices[1:]])
            xx2 = np.minimum(x2[current], x2[indices[1:]])
            yy2 = np.minimum(y2[current], y2[indices[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            union = areas[current] + areas[indices[1:]] - intersection
            iou = intersection / union
            
            # Remove boxes with high IoU
            indices = indices[1:][iou < threshold]
        
        return keep
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute YOLO loss (simplified)."""
        # Simplified loss computation
        # In practice, YOLO loss is complex with multiple components
        
        # Box regression loss (MSE)
        box_loss = np.mean((predictions[:, :4] - targets[:, :4]) ** 2)
        
        # Objectness loss (BCE)
        obj_loss = self._binary_cross_entropy(predictions[:, 4], targets[:, 4])
        
        # Classification loss (BCE for multi-label)
        class_loss = self._binary_cross_entropy(predictions[:, 5:], targets[:, 5:])
        
        # Weighted combination
        total_loss = 5.0 * box_loss + 1.0 * obj_loss + 1.0 * class_loss
        
        return total_loss
    
    def _binary_cross_entropy(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        pred = np.clip(pred, 1e-7, 1 - 1e-7)
        return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))


class YOLOv5(YOLOv4):
    """YOLOv5 implementation with Focus layer and improvements."""
    
    def __init__(self, num_classes: int = 80,
                 input_size: Tuple[int, int] = (640, 640),
                 model_size: str = 's'):
        """Initialize YOLOv5.
        
        Args:
            num_classes: Number of object classes
            input_size: Input image size (H, W)
            model_size: Model size ('s', 'm', 'l', 'x')
        """
        self.model_size = model_size
        self.depth_multiple, self.width_multiple = self._get_model_params(model_size)
        
        super().__init__(num_classes, input_size)
        self.name = f"YOLOv5{model_size}"
    
    def _get_model_params(self, size: str) -> Tuple[float, float]:
        """Get depth and width multipliers for different model sizes."""
        params = {
            's': (0.33, 0.50),
            'm': (0.67, 0.75),
            'l': (1.0, 1.0),
            'x': (1.33, 1.25)
        }
        return params.get(size, (0.33, 0.50))
    
    def _build_backbone(self):
        """Build YOLOv5 backbone with Focus layer."""
        # Focus layer (space-to-depth)
        self.add(FocusLayer(channels=int(64 * self.width_multiple)))
        
        # CSP Bottleneck layers
        self.add(Conv2D(int(128 * self.width_multiple), 3, stride=2, 
                       padding='same', activation='relu'))
        self.add(C3Block(int(128 * self.width_multiple), 
                        n_blocks=int(3 * self.depth_multiple)))
        
        self.add(Conv2D(int(256 * self.width_multiple), 3, stride=2,
                       padding='same', activation='relu'))
        self.add(C3Block(int(256 * self.width_multiple),
                        n_blocks=int(6 * self.depth_multiple)))
        
        self.add(Conv2D(int(512 * self.width_multiple), 3, stride=2,
                       padding='same', activation='relu'))
        self.add(C3Block(int(512 * self.width_multiple),
                        n_blocks=int(9 * self.depth_multiple)))
        
        self.add(Conv2D(int(1024 * self.width_multiple), 3, stride=2,
                       padding='same', activation='relu'))
        self.add(C3Block(int(1024 * self.width_multiple),
                        n_blocks=int(3 * self.depth_multiple)))
        
        self.add(SPPFLayer(int(1024 * self.width_multiple)))


class CSPBlock(BaseModel):
    """Cross Stage Partial Block."""
    
    def __init__(self, in_channels: int, out_channels: int, n_blocks: int = 1):
        super().__init__(name="CSPBlock")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self._build_block()
    
    def _build_block(self):
        # Split channels
        mid_channels = self.out_channels // 2
        
        # Main branch
        self.conv1 = Conv2D(mid_channels, 1, activation='relu')
        self.conv2 = Conv2D(mid_channels, 1, activation='relu')
        
        # Residual blocks
        self.blocks = []
        for _ in range(self.n_blocks):
            self.blocks.append(ResidualBlock(mid_channels))
        
        # Concatenation conv
        self.conv3 = Conv2D(self.out_channels, 1, activation='relu')
        self.bn = BatchNormalization()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        # Split
        x1 = self.conv1.forward(x)
        x2 = self.conv2.forward(x)
        
        # Apply residual blocks to x2
        for block in self.blocks:
            x2 = block.forward(x2)
        
        # Concatenate and output
        concat = np.concatenate([x1, x2], axis=1)
        output = self.conv3.forward(concat)
        output = self.bn.forward(output)
        
        self.output = output
        return output


class SPPBlock(BaseModel):
    """Spatial Pyramid Pooling Block."""
    
    def __init__(self, channels: int):
        super().__init__(name="SPPBlock")
        self.channels = channels
        self._build_block()
    
    def _build_block(self):
        self.conv1 = Conv2D(self.channels // 2, 1, activation='relu')
        self.conv2 = Conv2D(self.channels, 1, activation='relu')
        
        # Different pooling sizes
        self.pool1 = MaxPool2D(5, stride=1, padding='same')
        self.pool2 = MaxPool2D(9, stride=1, padding='same')
        self.pool3 = MaxPool2D(13, stride=1, padding='same')
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        x = self.conv1.forward(x)
        
        # Multiple pooling scales
        p1 = self.pool1.forward(x)
        p2 = self.pool2.forward(x)
        p3 = self.pool3.forward(x)
        
        # Concatenate all scales
        concat = np.concatenate([x, p1, p2, p3], axis=1)
        output = self.conv2.forward(concat)
        
        self.output = output
        return output


class FocusLayer(BaseModel):
    """Focus layer for YOLOv5 (space-to-depth)."""
    
    def __init__(self, channels: int):
        super().__init__(name="FocusLayer")
        self.channels = channels
        self.conv = Conv2D(channels, 3, padding='same', activation='relu')
        self.bn = BatchNormalization()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        # Space-to-depth: slice and concatenate
        N, C, H, W = x.shape
        
        # Sample every other pixel to create 4 sub-images
        x1 = x[:, :, ::2, ::2]  # Top-left
        x2 = x[:, :, 1::2, ::2]  # Top-right
        x3 = x[:, :, ::2, 1::2]  # Bottom-left
        x4 = x[:, :, 1::2, 1::2]  # Bottom-right
        
        # Concatenate along channel dimension
        concat = np.concatenate([x1, x2, x3, x4], axis=1)
        
        # Apply convolution
        output = self.conv.forward(concat)
        output = self.bn.forward(output)
        
        self.output = output
        return output


class C3Block(BaseModel):
    """C3 Block for YOLOv5."""
    
    def __init__(self, channels: int, n_blocks: int = 1):
        super().__init__(name="C3Block")
        self.channels = channels
        self.n_blocks = n_blocks
        self._build_block()
    
    def _build_block(self):
        self.conv1 = Conv2D(self.channels // 2, 1, activation='relu')
        self.conv2 = Conv2D(self.channels // 2, 1, activation='relu')
        self.conv3 = Conv2D(self.channels, 1, activation='relu')
        
        # Bottleneck blocks
        self.blocks = []
        for _ in range(self.n_blocks):
            block = BaseModel(name="Bottleneck")
            block.add(Conv2D(self.channels // 2, 1, activation='relu'))
            block.add(Conv2D(self.channels // 2, 3, padding='same', activation='relu'))
            self.blocks.append(block)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        x1 = self.conv1.forward(x)
        x2 = self.conv2.forward(x)
        
        # Apply bottleneck blocks
        for block in self.blocks:
            x2 = x2 + block.forward(x2)  # Residual connection
        
        # Concatenate and output
        concat = np.concatenate([x1, x2], axis=1)
        output = self.conv3.forward(concat)
        
        self.output = output
        return output


class SPPFLayer(BaseModel):
    """SPPF (Spatial Pyramid Pooling - Fast) Layer."""
    
    def __init__(self, channels: int):
        super().__init__(name="SPPFLayer")
        self.channels = channels
        self._build_block()
    
    def _build_block(self):
        self.conv1 = Conv2D(self.channels // 2, 1, activation='relu')
        self.conv2 = Conv2D(self.channels, 1, activation='relu')
        self.pool = MaxPool2D(5, stride=1, padding='same')
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        x = self.conv1.forward(x)
        
        # Apply pooling multiple times
        p1 = self.pool.forward(x)
        p2 = self.pool.forward(p1)
        p3 = self.pool.forward(p2)
        
        # Concatenate all
        concat = np.concatenate([x, p1, p2, p3], axis=1)
        output = self.conv2.forward(concat)
        
        self.output = output
        return output


class DetectionLayer(BaseModel):
    """YOLO detection layer."""
    
    def __init__(self, num_classes: int, num_anchors: int, channels: int):
        super().__init__(name="DetectionLayer")
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.channels = channels
        
        # Output channels: (x, y, w, h, conf, classes) * num_anchors
        output_channels = num_anchors * (5 + num_classes)
        self.conv = Conv2D(output_channels, 1, activation=None)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        output = self.conv.forward(x)
        self.output = output
        return output 