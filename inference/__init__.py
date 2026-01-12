"""
Clinical Inference Engine for Multi-Cancer Detection

Provides production-ready inference capabilities:
- Single-patient predictions with explanations
- Batch processing for clinical workflows
- Integration with XAI for clinical interpretability
- Structured output for clinical decision support
"""

from .predictor import CancerPredictor
from .inference_pipeline import InferencePipeline

__all__ = [
    'CancerPredictor',
    'InferencePipeline'
]