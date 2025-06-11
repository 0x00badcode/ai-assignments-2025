"""
MLP Implementation - A comprehensive Multi-Layer Perceptron library from scratch
Author: Implementation for Deep Learning Assignment
"""

from .core.model import MLP, StoppingCriteria
from .core.layers import Dense
from .core.activations import *
from .core.losses import *
from .optimizers import *
from .regularization import *
from .evaluation import *

__version__ = "1.0.0"
__author__ = "MLP Implementation Team"

__all__ = [
    'MLP', 'StoppingCriteria', 'Dense',
    'ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'Linear',
    'MeanSquaredError', 'CrossEntropy', 'BinaryCrossEntropy',
    'SGD', 'Momentum', 'RMSProp', 'Adam',
    'L1Regularization', 'L2Regularization', 'ElasticNetRegularization',
    'ClassificationEvaluator', 'RegressionEvaluator', 'ConfusionMatrix'
] 