"""
Multi-Cancer AI Model Architecture Module

Provides:
- Shared feature extractors (ResNet, EfficientNet)
- Cancer-specific classification heads
- Multi-task learning architecture
- Transfer learning utilities
"""

from .backbone import get_backbone
from .heads import CancerClassificationHead, MultiCancerHeads
from .multi_cancer_model import MultiCancerModel

__all__ = [
    'get_backbone',
    'CancerClassificationHead',
    'MultiCancerHeads',
    'MultiCancerModel'
]