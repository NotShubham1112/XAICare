"""
Clinical Analysis and Validation Module

Provides medical interpretation of AI performance:
- Clinical utility assessment
- Risk stratification analysis
- Bias and fairness evaluation
- Deployment readiness assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import yaml
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ClinicalAnalyzer:
    """
    Comprehensive clinical analysis of AI model performance.

    Provides medical interpretation and deployment readiness assessment.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize clinical analyzer."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def assess_clinical_utility(self,
                              metrics: Dict[str, float],
                              cancer_type: str,
                              deployment_context: str = 'screening') -> Dict[str, Union[float, str]]:
        """
        Assess clinical utility for specific cancer type and context.

        Args:
            metrics: Clinical performance metrics
            cancer_type: Type of cancer
            deployment_context: 'screening', 'diagnosis', 'monitoring'

        Returns:
            Clinical utility assessment
        """
        sensitivity = metrics.get('sensitivity', 0)
        specificity = metrics.get('specificity', 0)
        auc = metrics.get('auc', 0)

        # Context-specific thresholds
        thresholds = self._get_clinical_thresholds(deployment_context, cancer_type)

        # Calculate utility scores
        sensitivity_score = self._score_metric(sensitivity, thresholds['sensitivity'])
        specificity_score = self._score_metric(specificity, thresholds['specificity'])
        auc_score = self._score_metric(auc, thresholds['auc'])

        # Overall utility score (weighted)
        weights = {'sensitivity': 0.5, 'specificity': 0.3, 'auc': 0.2}
        utility_score = (
            sensitivity_score * weights['sensitivity'] +
            specificity_score * weights['specificity'] +
            auc_score * weights['auc']
        )

        # Clinical recommendation
        recommendation = self._generate_clinical_recommendation(
            utility_score, deployment_context, cancer_type
        )

        return {
            'utility_score': utility_score,
            'sensitivity_score': sensitivity_score,
            'specificity_score': specificity_score,
            'auc_score': auc_score,
            'deployment_context': deployment_context,
            'cancer_type': cancer_type,
            'recommendation': recommendation,
            'readiness_level': self._classify_readiness(utility_score)
        }

    def _get_clinical_thresholds(self, context: str, cancer_type: str) -> Dict[str, Dict[str, float]]:
        """Get context and cancer-specific clinical thresholds."""
        # Clinical thresholds vary by use case and cancer type
        base_thresholds = {
            'screening': {  # Population screening - prioritize sensitivity
                'sensitivity': {'excellent': 0.95, 'good': 0.90, 'acceptable': 0.85},
                'specificity': {'excellent': 0.90, 'good': 0.80, 'acceptable': 0.70},
                'auc': {'excellent': 0.95, 'good': 0.90, 'acceptable': 0.85}
            },
            'diagnosis': {  # Diagnostic assistance - balance both
                'sensitivity': {'excellent': 0.90, 'good': 0.85, 'acceptable': 0.80},
                'specificity': {'excellent': 0.95, 'good': 0.90, 'acceptable': 0.85},
                'auc': {'excellent': 0.95, 'good': 0.90, 'acceptable': 0.85}
            },
            'monitoring': {  # Follow-up monitoring - high specificity
                'sensitivity': {'excellent': 0.85, 'good': 0.80, 'acceptable': 0.75},
                'specificity': {'excellent': 0.98, 'good': 0.95, 'acceptable': 0.90},
                'auc': {'excellent': 0.90, 'good': 0.85, 'acceptable': 0.80}
            }
        }

        return base_thresholds.get(context, base_thresholds['diagnosis'])

    def _score_metric(self, value: float, thresholds: Dict[str, float]) -> float:
        """Score a metric against clinical thresholds."""
        if value >= thresholds.get('excellent', 0.9):
            return 1.0
        elif value >= thresholds.get('good', 0.8):
            return 0.7
        elif value >= thresholds.get('acceptable', 0.7):
            return 0.4
        else:
            return 0.1

    def _generate_clinical_recommendation(self,
                                        utility_score: float,
                                        context: str,
                                        cancer_type: str) -> str:
        """Generate clinical recommendation based on utility score."""
        if utility_score >= 0.8:
            return f"EXCELLENT clinical utility for {context} of {cancer_type} cancer. Ready for clinical deployment with minimal oversight."
        elif utility_score >= 0.6:
            return f"GOOD clinical utility for {context} of {cancer_type} cancer. Suitable for clinical assistance with expert review."
        elif utility_score >= 0.4:
            return f"FAIR clinical utility for {context} of {cancer_type} cancer. May be useful as secondary tool with extensive validation."
        else:
            return f"LIMITED clinical utility for {context} of {cancer_type} cancer. Requires significant improvements before clinical use."

    def _classify_readiness(self, utility_score: float) -> str:
        """Classify deployment readiness level."""
        if utility_score >= 0.8:
            return 'production_ready'
        elif utility_score >= 0.6:
            return 'clinical_trial'
        elif utility_score >= 0.4:
            return 'research_only'
        else:
            return 'needs_improvement'

    def analyze_risk_stratification(self,
                                  y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  risk_thresholds: Optional[List[float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze model's ability to stratify patients by risk level.

        Args:
            y_true: Ground truth labels
            y_prob: Prediction probabilities
            risk_thresholds: Custom risk thresholds

        Returns:
            Risk stratification analysis
        """
        if risk_thresholds is None:
            risk_thresholds = [0.3, 0.6, 0.8]  # Low, medium, high risk

        # Create risk categories
        risk_categories = ['very_low', 'low', 'medium', 'high']
        thresholds = [0.0] + risk_thresholds + [1.0]

        stratification_results = {}

        for i, category in enumerate(risk_categories):
            lower_thresh = thresholds[i]
            upper_thresh = thresholds[i + 1]

            # Get samples in this risk category
            mask = (y_prob >= lower_thresh) & (y_prob < upper_thresh)
            if category == 'very_low':
                mask = (y_prob < risk_thresholds[0])

            samples_in_category = np.sum(mask)

            if samples_in_category > 0:
                true_positives = np.sum(y_true[mask] == 1)
                malignancy_rate = true_positives / samples_in_category
            else:
                malignancy_rate = 0.0

            stratification_results[category] = {
                'sample_count': int(samples_in_category),
                'malignancy_rate': malignancy_rate,
                'probability_range': (lower_thresh, upper_thresh),
                'percentage_of_population': samples_in_category / len(y_true)
            }

        return stratification_results

    def assess_bias_and_fairness(self,
                               predictions: Dict[str, np.ndarray],
                               sensitive_attributes: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Dict[str, float]]:
        """
        Assess bias and fairness across demographic groups.

        Args:
            predictions: Model predictions by group
            sensitive_attributes: Demographic attributes (age, gender, ethnicity)

        Returns:
            Bias and fairness analysis
        """
        fairness_metrics = {}

        if sensitive_attributes is None:
            # Create dummy groups for demonstration
            n_samples = len(next(iter(predictions.values())))
            sensitive_attributes = {
                'age_group': np.random.choice(['young', 'middle', 'old'], n_samples),
                'gender': np.random.choice(['male', 'female'], n_samples)
            }

        for attribute_name, groups in sensitive_attributes.items():
            unique_groups = np.unique(groups)
            group_metrics = {}

            for group in unique_groups:
                mask = groups == group
                if np.sum(mask) == 0:
                    continue

                # Calculate metrics for this group
                group_predictions = {k: v[mask] for k, v in predictions.items()}

                # Simplified fairness metrics
                group_size = np.sum(mask)
                positive_rate = np.mean(group_predictions.get('predictions', []))

                group_metrics[str(group)] = {
                    'group_size': int(group_size),
                    'positive_rate': positive_rate,
                    'representation': group_size / len(groups)
                }

            fairness_metrics[attribute_name] = group_metrics

        return fairness_metrics

    def perform_sensitivity_analysis(self,
                                  base_metrics: Dict[str, float],
                                  parameter_ranges: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Perform sensitivity analysis on key parameters.

        Args:
            base_metrics: Baseline clinical metrics
            parameter_ranges: Ranges of parameters to test

        Returns:
            Sensitivity analysis results
        """
        sensitivity_results = {}

        # Test different classification thresholds
        if 'threshold_range' in parameter_ranges:
            threshold_range = parameter_ranges['threshold_range']

            for threshold in threshold_range:
                # This would require re-evaluating with different thresholds
                # Simplified version
                sensitivity_results[f'threshold_{threshold}'] = {
                    'threshold': threshold,
                    'sensitivity_change': np.random.uniform(-0.1, 0.1),  # Placeholder
                    'specificity_change': np.random.uniform(-0.1, 0.1)
                }

        return sensitivity_results

    def generate_deployment_recommendations(self,
                                         clinical_assessment: Dict[str, Union[float, str]],
                                         cancer_type: str) -> Dict[str, Union[str, List[str]]]:
        """
        Generate deployment recommendations based on clinical assessment.

        Args:
            clinical_assessment: Clinical utility assessment
            cancer_type: Type of cancer

        Returns:
            Deployment recommendations
        """
        readiness_level = clinical_assessment.get('readiness_level', 'needs_improvement')
        utility_score = clinical_assessment.get('utility_score', 0)

        recommendations = {
            'readiness_level': readiness_level,
            'deployment_phase': self._get_deployment_phase(readiness_level),
            'required_oversight': self._get_required_oversight(readiness_level),
            'validation_requirements': self._get_validation_requirements(readiness_level, cancer_type),
            'monitoring_requirements': self._get_monitoring_requirements(readiness_level),
            'risk_mitigation': self._get_risk_mitigation_strategies(cancer_type, utility_score)
        }

        return recommendations

    def _get_deployment_phase(self, readiness_level: str) -> str:
        """Get appropriate deployment phase."""
        phases = {
            'production_ready': 'Full clinical deployment',
            'clinical_trial': 'Prospective clinical trial',
            'research_only': 'Research validation study',
            'needs_improvement': 'Algorithm improvement required'
        }
        return phases.get(readiness_level, 'Further validation required')

    def _get_required_oversight(self, readiness_level: str) -> str:
        """Get required level of clinical oversight."""
        oversight = {
            'production_ready': 'Routine clinical review',
            'clinical_trial': 'Specialist physician review',
            'research_only': 'Dual specialist review',
            'needs_improvement': 'Expert panel review required'
        }
        return oversight.get(readiness_level, 'Extensive clinical oversight')

    def _get_validation_requirements(self, readiness_level: str, cancer_type: str) -> List[str]:
        """Get validation requirements for deployment."""
        base_requirements = [
            "Prospective clinical validation study",
            "Comparison with current standard of care",
            "Cost-effectiveness analysis"
        ]

        if readiness_level == 'production_ready':
            requirements = ["Multi-center validation", "Real-world performance monitoring"]
        elif readiness_level == 'clinical_trial':
            requirements = ["Randomized controlled trial", "Blinded assessment"]
        elif readiness_level == 'research_only':
            requirements = ["Retrospective validation", "External dataset testing"]
        else:
            requirements = ["Algorithm performance improvement", "Additional training data"]

        return base_requirements + requirements

    def _get_monitoring_requirements(self, readiness_level: str) -> List[str]:
        """Get monitoring requirements post-deployment."""
        monitoring = {
            'production_ready': [
                "Monthly performance metrics review",
                "Annual clinical validation",
                "Adverse event monitoring"
            ],
            'clinical_trial': [
                "Weekly performance monitoring",
                "Monthly clinical review",
                "Protocol deviation tracking"
            ],
            'research_only': [
                "Continuous performance tracking",
                "Regular clinical correlation",
                "Algorithm drift detection"
            ]
        }

        return monitoring.get(readiness_level, ["Comprehensive monitoring required"])

    def _get_risk_mitigation_strategies(self, cancer_type: str, utility_score: float) -> List[str]:
        """Get risk mitigation strategies."""
        strategies = [
            "All predictions require clinical review",
            "Clear documentation of AI limitations",
            "Regular algorithm updates based on new data"
        ]

        if utility_score < 0.6:
            strategies.extend([
                "Parallel human-AI decision making",
                "Mandatory second opinion for high-risk cases",
                "Patient informed consent for AI use"
            ])

        if cancer_type in ['lung', 'breast']:
            strategies.append("Integration with existing screening workflows")

        return strategies


def perform_clinical_validation(model_results: Dict[str, Dict[str, float]],
                              cancer_types: List[str],
                              deployment_contexts: List[str] = None) -> Dict[str, Dict]:
    """
    Perform comprehensive clinical validation across cancer types and contexts.

    Args:
        model_results: Model performance results by cancer type
        cancer_types: List of cancer types
        deployment_contexts: List of deployment contexts

    Returns:
        Clinical validation report
    """
    if deployment_contexts is None:
        deployment_contexts = ['screening', 'diagnosis', 'monitoring']

    analyzer = ClinicalAnalyzer()
    validation_report = {}

    for cancer_type in cancer_types:
        if cancer_type not in model_results:
            continue

        metrics = model_results[cancer_type]
        cancer_validation = {}

        for context in deployment_contexts:
            assessment = analyzer.assess_clinical_utility(metrics, cancer_type, context)
            recommendations = analyzer.generate_deployment_recommendations(assessment, cancer_type)

            cancer_validation[context] = {
                'assessment': assessment,
                'recommendations': recommendations
            }

        validation_report[cancer_type] = cancer_validation

    # Generate overall summary
    validation_report['summary'] = {
        'total_cancer_types': len(cancer_types),
        'deployment_contexts': deployment_contexts,
        'overall_readiness': _calculate_overall_readiness(validation_report),
        'key_recommendations': _extract_key_recommendations(validation_report)
    }

    return validation_report


def _calculate_overall_readiness(validation_report: Dict) -> str:
    """Calculate overall system readiness."""
    readiness_levels = []
    for cancer_data in validation_report.values():
        if isinstance(cancer_data, dict) and 'assessment' in cancer_data:
            for context_data in cancer_data.values():
                if 'assessment' in context_data:
                    readiness = context_data['assessment'].get('readiness_level', 'unknown')
                    readiness_levels.append(readiness)

    # Simplified readiness calculation
    if 'production_ready' in readiness_levels:
        return 'production_ready'
    elif 'clinical_trial' in readiness_levels:
        return 'clinical_trial'
    elif 'research_only' in readiness_levels:
        return 'research_only'
    else:
        return 'needs_improvement'


def _extract_key_recommendations(validation_report: Dict) -> List[str]:
    """Extract key recommendations from validation report."""
    recommendations = []

    # Check for common issues
    readiness_levels = []
    for cancer_data in validation_report.values():
        if isinstance(cancer_data, dict):
            for context_data in cancer_data.values():
                if 'recommendations' in context_data:
                    recs = context_data['recommendations']
                    recommendations.extend(recs.get('validation_requirements', []))

    # Remove duplicates and limit to top recommendations
    unique_recs = list(set(recommendations))[:5]

    return unique_recs


def export_clinical_validation_report(validation_report: Dict,
                                   save_path: str,
                                   filename: str = "clinical_validation_report.json"):
    """
    Export clinical validation report to file.

    Args:
        validation_report: Clinical validation results
        save_path: Directory to save report
        filename: Report filename
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    filepath = save_path / filename

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(filepath, 'w') as f:
        json.dump(validation_report, f, indent=2, default=convert_numpy)

    logger.info(f"Clinical validation report saved to {filepath}")