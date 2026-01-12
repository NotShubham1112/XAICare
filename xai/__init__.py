"""
Explainable AI Module for Multi-Cancer Detection

Provides visual and textual explanations for model predictions:
- Grad-CAM for localization
- Saliency maps for pixel importance
- Textual explanations for clinical interpretation
"""

from .grad_cam import GradCAM, GradCAMPlusPlus
from .saliency import SaliencyMapGenerator
from .explanations import ExplanationGenerator

__all__ = [
    'GradCAM',
    'GradCAMPlusPlus',
    'SaliencyMapGenerator',
    'ExplanationGenerator'
]