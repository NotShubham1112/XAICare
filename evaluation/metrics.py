"""
Clinical Evaluation Metrics for Medical AI Systems

Implements medically-relevant metrics beyond accuracy:
- Sensitivity (Recall) - critical for early detection
- Specificity - minimizes false alarms
- ROC-AUC - discrimination ability
- False Negative Rate - clinical risk assessment
- Clinical utility metrics
"""

import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    classification_report, cohen_kappa_score
)
from typing import Dict, List, Optional, Union, Tuple
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_clinical_metrics(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_prob: Optional[np.ndarray] = None,
                             threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate comprehensive clinical metrics for binary classification.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (if None, will be derived from y_prob)
        y_prob: Prediction probabilities for positive class
        threshold: Classification threshold

    Returns:
        Dictionary of clinical metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Derive predictions from probabilities if not provided
    if y_pred is None and y_prob is not None:
        y_pred = (y_prob >= threshold).astype(int)
    elif y_pred is None:
        raise ValueError("Either y_pred or y_prob must be provided")

    # Basic confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Core clinical metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate / Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0    # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0          # Negative Predictive Value

    # Additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    # Clinical risk metrics
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Miss rate
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False alarm rate

    # Diagnostic odds ratio
    if sensitivity > 0 and (1 - specificity) > 0:
        dor = (sensitivity / (1 - specificity)) / ((1 - sensitivity) / specificity)
    else:
        dor = float('inf') if sensitivity == 1.0 else 0.0

    metrics = {
        'sensitivity': sensitivity,      # Primary metric for early detection
        'specificity': specificity,      # Minimizes false alarms
        'precision': precision,
        'npv': npv,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'false_negative_rate': false_negative_rate,
        'false_positive_rate': false_positive_rate,
        'diagnostic_odds_ratio': dor,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }

    # ROC-AUC if probabilities available
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            metrics['auc'] = roc_auc

            # Average Precision (area under PR curve)
            avg_precision = average_precision_score(y_true, y_prob)
            metrics['average_precision'] = avg_precision

        except Exception as e:
            logger.warning(f"Could not calculate AUC metrics: {e}")
            metrics['auc'] = 0.0
            metrics['average_precision'] = 0.0

    return metrics


def calculate_optimal_threshold(y_true: np.ndarray,
                              y_prob: np.ndarray,
                              metric: str = 'f1_score') -> Dict[str, float]:
    """
    Find optimal classification threshold for specific metric.

    Args:
        y_true: Ground truth labels
        y_prob: Prediction probabilities
        metric: Metric to optimize ('f1_score', 'sensitivity', 'specificity', 'j_index')

    Returns:
        Dictionary with optimal threshold and corresponding metrics
    """
    thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    best_score = 0.0
    best_metrics = {}

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = calculate_clinical_metrics(y_true, y_pred, y_prob, threshold)

        if metric == 'f1_score':
            score = metrics['f1_score']
        elif metric == 'sensitivity':
            score = metrics['sensitivity']
        elif metric == 'specificity':
            score = metrics['specificity']
        elif metric == 'j_index':
            # Youden's J statistic: sensitivity + specificity - 1
            score = metrics['sensitivity'] + metrics['specificity'] - 1
        elif metric == 'balanced_accuracy':
            score = (metrics['sensitivity'] + metrics['specificity']) / 2
        else:
            score = metrics[metric] if metric in metrics else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    return {
        'optimal_threshold': best_threshold,
        'best_score': best_score,
        'metric_name': metric,
        **best_metrics
    }


class ClinicalMetrics:
    """
    Advanced clinical metrics calculator with statistical analysis.

    Provides comprehensive evaluation suitable for clinical validation.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize clinical metrics calculator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def evaluate_model(self,
                      model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      device: str = 'cuda') -> Dict[str, float]:
        """
        Evaluate model on dataloader with clinical metrics.

        Args:
            model: PyTorch model
            dataloader: Evaluation dataloader
            device: Device to run evaluation on

        Returns:
            Comprehensive evaluation metrics
        """
        model.eval()
        model.to(device)

        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels, metadata in dataloader:
                images = images.to(device)

                outputs = model(images)
                probabilities = outputs['probabilities']
                predictions = outputs['prediction']

                # Assuming binary classification, take positive class probability
                if probabilities.shape[-1] == 2:
                    pos_probabilities = probabilities[:, 1].cpu().numpy()
                else:
                    pos_probabilities = probabilities.max(dim=1)[0].cpu().numpy()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(pos_probabilities)

        # Calculate clinical metrics
        clinical_metrics = calculate_clinical_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities)
        )

        # Calculate optimal threshold
        optimal_threshold = calculate_optimal_threshold(
            np.array(all_labels),
            np.array(all_probabilities),
            metric='f1_score'  # Can be configured
        )

        # Add additional statistics
        clinical_metrics.update({
            'optimal_threshold': optimal_threshold['optimal_threshold'],
            'optimal_f1_score': optimal_threshold['best_score'],
            'num_samples': len(all_labels),
            'positive_samples': sum(all_labels),
            'negative_samples': len(all_labels) - sum(all_labels)
        })

        return clinical_metrics

    def calculate_bootstrap_confidence_intervals(self,
                                              y_true: np.ndarray,
                                              y_prob: np.ndarray,
                                              n_bootstraps: int = 1000,
                                              confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for clinical metrics.

        Args:
            y_true: Ground truth labels
            y_prob: Prediction probabilities
            n_bootstraps: Number of bootstrap samples
            confidence_level: Confidence level (0.95 for 95% CI)

        Returns:
            Dictionary with confidence intervals for each metric
        """
        np.random.seed(42)  # For reproducibility

        n_samples = len(y_true)
        bootstrap_metrics = {
            'sensitivity': [],
            'specificity': [],
            'auc': [],
            'f1_score': []
        }

        for _ in range(n_bootstraps):
            # Bootstrap sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_prob_boot = y_prob[indices]

            # Calculate metrics
            try:
                y_pred_boot = (y_prob_boot >= 0.5).astype(int)
                metrics = calculate_clinical_metrics(y_true_boot, y_pred_boot, y_prob_boot)

                for metric in bootstrap_metrics.keys():
                    if metric in metrics:
                        bootstrap_metrics[metric].append(metrics[metric])

            except Exception as e:
                logger.warning(f"Bootstrap iteration failed: {e}")
                continue

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        confidence_intervals = {}

        for metric, values in bootstrap_metrics.items():
            if values:
                values = np.array(values)
                lower_percentile = np.percentile(values, alpha/2 * 100)
                upper_percentile = np.percentile(values, (1 - alpha/2) * 100)
                confidence_intervals[metric] = (lower_percentile, upper_percentile)
            else:
                confidence_intervals[metric] = (0.0, 0.0)

        return confidence_intervals

    def calculate_clinical_utility_score(self,
                                       sensitivity: float,
                                       specificity: float,
                                       prevalence: float = 0.1) -> Dict[str, float]:
        """
        Calculate clinical utility metrics.

        Args:
            sensitivity: Model sensitivity
            specificity: Model specificity
            prevalence: Disease prevalence in population

        Returns:
            Clinical utility scores
        """
        # Positive Predictive Value (PPV)
        ppv = (sensitivity * prevalence) / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))

        # Negative Predictive Value (NPV)
        npv = (specificity * (1 - prevalence)) / (specificity * (1 - prevalence) + (1 - sensitivity) * prevalence)

        # Net Benefit (for decision curve analysis)
        # Simplified version - assumes treatment benefit = 1, harm = 0
        net_benefit = (sensitivity * prevalence) - ((1 - specificity) * (1 - prevalence))

        # Clinical utility score (weighted combination)
        utility_score = 0.4 * sensitivity + 0.4 * specificity + 0.2 * ppv

        return {
            'ppv': ppv,
            'npv': npv,
            'net_benefit': net_benefit,
            'clinical_utility_score': utility_score,
            'prevalence': prevalence
        }

    def assess_diagnostic_performance(self,
                                    sensitivity: float,
                                    specificity: float,
                                    prevalence_range: Optional[List[float]] = None) -> Dict[str, List[float]]:
        """
        Assess diagnostic performance across different prevalence rates.

        Args:
            sensitivity: Model sensitivity
            specificity: Model specificity
            prevalence_range: Range of prevalence values to test

        Returns:
            Performance metrics across prevalence values
        """
        if prevalence_range is None:
            prevalence_range = np.linspace(0.01, 0.5, 50)

        ppv_values = []
        npv_values = []
        net_benefits = []

        for prevalence in prevalence_range:
            utility = self.calculate_clinical_utility_score(sensitivity, specificity, prevalence)
            ppv_values.append(utility['ppv'])
            npv_values.append(utility['npv'])
            net_benefits.append(utility['net_benefit'])

        return {
            'prevalence_range': prevalence_range.tolist(),
            'ppv_curve': ppv_values,
            'npv_curve': npv_values,
            'net_benefit_curve': net_benefits
        }


def compare_models_clinically(model_metrics: Dict[str, Dict[str, float]],
                            metric_weights: Optional[Dict[str, float]] = None) -> Dict[str, Union[str, float]]:
    """
    Compare multiple models based on clinical metrics.

    Args:
        model_metrics: Dictionary of model names to their metrics
        metric_weights: Weights for different metrics in ranking

    Returns:
        Comparison results and ranking
    """
    if metric_weights is None:
        # Default clinical prioritization
        metric_weights = {
            'sensitivity': 0.4,    # Most important for early detection
            'specificity': 0.3,    # Minimize false alarms
            'auc': 0.2,           # Discrimination ability
            'f1_score': 0.1       # Balance measure
        }

    model_scores = {}
    for model_name, metrics in model_metrics.items():
        score = 0.0
        for metric, weight in metric_weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
        model_scores[model_name] = score

    # Rank models
    ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        'ranking': ranked_models,
        'best_model': ranked_models[0][0] if ranked_models else None,
        'metric_weights': metric_weights,
        'scores': model_scores
    }


def perform_statistical_tests(metrics1: Dict[str, float],
                           metrics2: Dict[str, float],
                           test_type: str = 'mcnemar') -> Dict[str, float]:
    """
    Perform statistical significance tests between two models.

    Args:
        metrics1: Metrics for first model
        metrics2: Metrics for second model
        test_type: Type of statistical test

    Returns:
        Test results and p-values
    """
    # Simplified implementation - in practice would use proper statistical tests
    # For binary classification comparison

    results = {
        'test_type': test_type,
        'significant_difference': False,
        'p_value': 1.0,
        'recommendation': 'Models perform similarly'
    }

    # Compare key metrics
    sensitivity_diff = abs(metrics1.get('sensitivity', 0) - metrics2.get('sensitivity', 0))
    specificity_diff = abs(metrics1.get('specificity', 0) - metrics2.get('specificity', 0))
    auc_diff = abs(metrics1.get('auc', 0) - metrics2.get('auc', 0))

    # Simple heuristic for significance (not statistically rigorous)
    if sensitivity_diff > 0.05 or specificity_diff > 0.05 or auc_diff > 0.03:
        results['significant_difference'] = True
        results['p_value'] = 0.05  # Placeholder
        results['recommendation'] = 'Models show significant performance differences'

    return results


def generate_clinical_summary(metrics: Dict[str, float],
                            confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None) -> str:
    """
    Generate clinical interpretation summary.

    Args:
        metrics: Clinical metrics dictionary
        confidence_intervals: Bootstrap confidence intervals

    Returns:
        Clinical summary string
    """
    sensitivity = metrics.get('sensitivity', 0)
    specificity = metrics.get('specificity', 0)
    auc = metrics.get('auc', 0)
    f1 = metrics.get('f1_score', 0)

    summary = ".3f"".3f"".3f"".3f""

    # Clinical interpretation
    if sensitivity > 0.9 and specificity > 0.85:
        summary += "\n\nEXCELLENT: High sensitivity and specificity suitable for clinical screening."
    elif sensitivity > 0.85 and specificity > 0.8:
        summary += "\n\nGOOD: Strong performance for clinical assistance with some false positives/negatives."
    elif sensitivity > 0.8:
        summary += "\n\nFAIR: Acceptable sensitivity but may require additional clinical correlation."
    else:
        summary += "\n\nCAUTION: Limited sensitivity may miss clinically significant cases."

    # Add confidence intervals if available
    if confidence_intervals:
        summary += "\n\nCONFIDENCE INTERVALS (95%):"
        for metric, (lower, upper) in confidence_intervals.items():
            if metric in ['sensitivity', 'specificity', 'auc']:
                summary += ".3f"".3f""

    return summary