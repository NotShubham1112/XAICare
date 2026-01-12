"""
Clinical Evaluation Module for Medical AI Systems

Provides comprehensive evaluation metrics suitable for clinical deployment:
- Sensitivity, specificity, ROC-AUC analysis
- Clinical utility assessment
- Bias and fairness evaluation
- Statistical significance testing
"""

from .metrics import calculate_clinical_metrics, ClinicalMetrics
from .clinical_analysis import ClinicalAnalyzer, perform_clinical_validation
from .plots import EvaluationPlots, create_evaluation_report

__all__ = [
    'calculate_clinical_metrics',
    'ClinicalMetrics',
    'ClinicalAnalyzer',
    'perform_clinical_validation',
    'EvaluationPlots',
    'create_evaluation_report'
]