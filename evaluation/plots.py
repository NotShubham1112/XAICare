"""
Clinical Evaluation Visualization Module

Provides comprehensive plotting functions for:
- ROC curves and AUC analysis
- Clinical metric comparison plots
- Bias and fairness visualizations
- Deployment readiness dashboards
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import Dict, List, Optional, Union, Tuple
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EvaluationPlots:
    """
    Clinical evaluation visualization tools.

    Provides publication-quality plots for medical AI evaluation.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize plotting utilities."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up plotting parameters
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12

    def plot_roc_curve(self,
                      y_true: np.ndarray,
                      y_prob: np.ndarray,
                      title: str = "ROC Curve",
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve with AUC.

        Args:
            y_true: Ground truth labels
            y_prob: Prediction probabilities
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label='.3f')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        # Add optimal threshold point
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro',
                label='.3f')
        ax.legend(loc="lower right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")

        return fig

    def plot_precision_recall_curve(self,
                                  y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  title: str = "Precision-Recall Curve",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot precision-recall curve.

        Args:
            y_true: Ground truth labels
            y_prob: Prediction probabilities
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        avg_precision = np.mean(precision)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(recall, precision, color='blue', lw=2,
                label='.3f')
        ax.axhline(y=np.sum(y_true) / len(y_true), color='red', linestyle='--',
                  label='Baseline (Prevalence)')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curve saved to {save_path}")

        return fig

    def plot_clinical_metrics_radar(self,
                                  metrics: Dict[str, float],
                                  title: str = "Clinical Metrics Overview",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot clinical metrics on a radar chart.

        Args:
            metrics: Dictionary of clinical metrics
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        # Key clinical metrics for radar plot
        radar_metrics = ['sensitivity', 'specificity', 'precision', 'auc', 'f1_score']
        values = []

        for metric in radar_metrics:
            if metric in metrics:
                values.append(metrics[metric])
            else:
                values.append(0.0)

        # Close the polygon
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics) + 1, endpoint=True)

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        ax.plot(angles, values, 'o-', linewidth=2, label='Model Performance')
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in radar_metrics])
        ax.set_ylim(0, 1)
        ax.set_title(title, size=16, fontweight='bold', pad=20)
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Clinical metrics radar plot saved to {save_path}")

        return fig

    def plot_threshold_analysis(self,
                              y_true: np.ndarray,
                              y_prob: np.ndarray,
                              title: str = "Threshold Analysis",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot metrics vs classification threshold.

        Args:
            y_true: Ground truth labels
            y_prob: Prediction probabilities
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        thresholds = np.linspace(0.01, 0.99, 99)

        sensitivities = []
        specificities = []
        f1_scores = []
        ppvs = []  # Positive Predictive Values

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)

            # Calculate metrics
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

            sensitivities.append(sensitivity)
            specificities.append(specificity)
            f1_scores.append(f1)
            ppvs.append(precision)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Sensitivity vs Threshold
        ax1.plot(thresholds, sensitivities, 'b-', linewidth=2, label='Sensitivity')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Sensitivity')
        ax1.set_title('Sensitivity vs Threshold')
        ax1.grid(True, alpha=0.3)

        # Specificity vs Threshold
        ax2.plot(thresholds, specificities, 'r-', linewidth=2, label='Specificity')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Specificity')
        ax2.set_title('Specificity vs Threshold')
        ax2.grid(True, alpha=0.3)

        # F1-Score vs Threshold
        ax3.plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1-Score')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('F1-Score')
        ax3.set_title('F1-Score vs Threshold')
        ax3.grid(True, alpha=0.3)

        # PPV vs Threshold
        ax4.plot(thresholds, ppvs, 'purple', linewidth=2, label='PPV')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Positive Predictive Value')
        ax4.set_title('PPV vs Threshold')
        ax4.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threshold analysis plot saved to {save_path}")

        return fig

    def plot_model_comparison(self,
                            model_results: Dict[str, Dict[str, float]],
                            metrics: List[str] = None,
                            title: str = "Model Comparison",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare multiple models across clinical metrics.

        Args:
            model_results: Dictionary of model results
            metrics: List of metrics to compare
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['sensitivity', 'specificity', 'auc', 'f1_score']

        # Prepare data
        model_names = list(model_results.keys())
        data = []

        for model_name in model_names:
            for metric in metrics:
                value = model_results[model_name].get(metric, 0)
                data.append({
                    'Model': model_name,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': value
                })

        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.barplot(data=df, x='Metric', y='Value', hue='Model', ax=ax)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Score')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")

        return fig

    def plot_confusion_matrix_heatmap(self,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    title: str = "Confusion Matrix",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix as a heatmap.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix heatmap saved to {save_path}")

        return fig

    def plot_risk_stratification(self,
                               risk_analysis: Dict[str, Dict[str, float]],
                               title: str = "Risk Stratification Analysis",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot risk stratification results.

        Args:
            risk_analysis: Risk stratification analysis results
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        categories = list(risk_analysis.keys())
        malignancy_rates = [risk_analysis[cat]['malignancy_rate'] for cat in categories]
        sample_counts = [risk_analysis[cat]['sample_count'] for cat in categories]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Malignancy rates by risk category
        bars1 = ax1.bar(categories, malignancy_rates, color='lightcoral', alpha=0.7)
        ax1.set_xlabel('Risk Category')
        ax1.set_ylabel('Malignancy Rate')
        ax1.set_title('Malignancy Rate by Risk Category')
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, rate in zip(bars1, malignancy_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    '.3f', ha='center', va='bottom')

        # Sample distribution
        bars2 = ax2.bar(categories, sample_counts, color='lightblue', alpha=0.7)
        ax2.set_xlabel('Risk Category')
        ax2.set_ylabel('Sample Count')
        ax2.set_title('Sample Distribution by Risk Category')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, count in zip(bars2, sample_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_counts) * 0.01,
                    str(int(count)), ha='center', va='bottom')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Risk stratification plot saved to {save_path}")

        return fig

    def plot_deployment_readiness_dashboard(self,
                                         clinical_assessment: Dict[str, Union[float, str]],
                                         cancer_type: str,
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create deployment readiness dashboard.

        Args:
            clinical_assessment: Clinical utility assessment
            cancer_type: Type of cancer
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Utility Score Gauge
        utility_score = clinical_assessment.get('utility_score', 0)

        # Create gauge chart
        self._create_gauge_chart(ax1, utility_score, "Clinical Utility Score",
                                ["Poor", "Fair", "Good", "Excellent"])

        # Component Scores
        components = ['sensitivity_score', 'specificity_score', 'auc_score']
        component_names = ['Sensitivity', 'Specificity', 'AUC']
        scores = [clinical_assessment.get(comp, 0) for comp in components]

        bars = ax2.bar(component_names, scores, color=['lightblue', 'lightgreen', 'lightcoral'])
        ax2.set_ylabel('Score')
        ax2.set_title('Component Scores')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, score in zip(bars, scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    '.3f', ha='center', va='bottom')

        # Readiness Level
        readiness_level = clinical_assessment.get('readiness_level', 'unknown')
        readiness_map = {
            'production_ready': 4,
            'clinical_trial': 3,
            'research_only': 2,
            'needs_improvement': 1
        }

        readiness_value = readiness_map.get(readiness_level, 1)
        readiness_labels = ['Needs\nImprovement', 'Research\nOnly', 'Clinical\nTrial', 'Production\nReady']

        ax3.barh(['Readiness'], [readiness_value], color='skyblue', height=0.5)
        ax3.set_xlim(0, 4)
        ax3.set_xticks(range(1, 5))
        ax3.set_xticklabels(readiness_labels, rotation=45, ha='right')
        ax3.set_title('Deployment Readiness Level')

        # Key Metrics Summary
        metrics_text = ".3f"".3f"".3f""

        ax4.text(0.1, 0.8, f"Cancer Type: {cancer_type}", fontsize=14, fontweight='bold')
        ax4.text(0.1, 0.6, f"Readiness: {readiness_level.replace('_', ' ').title()}", fontsize=12)
        ax4.text(0.1, 0.4, metrics_text, fontsize=10, verticalalignment='top')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Assessment Summary')

        plt.suptitle(f'Deployment Readiness Dashboard - {cancer_type.title()} Cancer',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Deployment readiness dashboard saved to {save_path}")

        return fig

    def _create_gauge_chart(self, ax, value, title, labels):
        """Create a simple gauge chart."""
        # Create semicircle
        theta = np.linspace(np.pi, 0, 100)
        x = np.cos(theta)
        y = np.sin(theta)

        # Color zones
        colors = ['red', 'orange', 'yellow', 'green']
        for i in range(4):
            start_angle = np.pi - (i * np.pi / 4)
            end_angle = np.pi - ((i + 1) * np.pi / 4)
            theta_zone = np.linspace(start_angle, end_angle, 25)
            x_zone = np.cos(theta_zone)
            y_zone = np.sin(theta_zone)
            ax.fill_between(x_zone, 0, y_zone, color=colors[i], alpha=0.3)

        # Value indicator
        value_angle = np.pi - (value * np.pi)
        x_indicator = np.cos(value_angle)
        y_indicator = np.sin(value_angle)
        ax.plot([0, x_indicator], [0, y_indicator], 'k-', linewidth=3)
        ax.plot(x_indicator, y_indicator, 'ko', markersize=8)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title)

        # Add labels
        for i, label in enumerate(labels):
            angle = np.pi - (i + 0.5) * (np.pi / 4)
            x_text = 0.8 * np.cos(angle)
            y_text = 0.8 * np.sin(angle)
            ax.text(x_text, y_text, label, ha='center', va='center', fontsize=8)


def create_evaluation_report(model_metrics: Dict[str, float],
                           plots_dir: str = "evaluation_plots",
                           report_title: str = "Clinical Evaluation Report") -> Dict[str, str]:
    """
    Create comprehensive evaluation report with all plots.

    Args:
        model_metrics: Clinical metrics dictionary
        plots_dir: Directory to save plots
        report_title: Title for the report

    Returns:
        Dictionary mapping plot types to file paths
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plotter = EvaluationPlots()
    saved_plots = {}

    try:
        # ROC Curve
        if 'auc' in model_metrics and 'y_true' in model_metrics and 'y_prob' in model_metrics:
            fig = plotter.plot_roc_curve(
                model_metrics['y_true'],
                model_metrics['y_prob'],
                f"{report_title} - ROC Curve",
                plots_dir / "roc_curve.png"
            )
            saved_plots['roc_curve'] = str(plots_dir / "roc_curve.png")
            plt.close(fig)

        # Clinical Metrics Radar
        fig = plotter.plot_clinical_metrics_radar(
            model_metrics,
            f"{report_title} - Clinical Metrics",
            plots_dir / "clinical_metrics_radar.png"
        )
        saved_plots['clinical_metrics_radar'] = str(plots_dir / "clinical_metrics_radar.png")
        plt.close(fig)

        # Threshold Analysis
        if 'y_true' in model_metrics and 'y_prob' in model_metrics:
            fig = plotter.plot_threshold_analysis(
                model_metrics['y_true'],
                model_metrics['y_prob'],
                f"{report_title} - Threshold Analysis",
                plots_dir / "threshold_analysis.png"
            )
            saved_plots['threshold_analysis'] = str(plots_dir / "threshold_analysis.png")
            plt.close(fig)

        # Confusion Matrix
        if 'y_true' in model_metrics and 'y_pred' in model_metrics:
            fig = plotter.plot_confusion_matrix_heatmap(
                model_metrics['y_true'],
                model_metrics['y_pred'],
                f"{report_title} - Confusion Matrix",
                plots_dir / "confusion_matrix.png"
            )
            saved_plots['confusion_matrix'] = str(plots_dir / "confusion_matrix.png")
            plt.close(fig)

    except Exception as e:
        logger.warning(f"Error creating evaluation plots: {e}")

    logger.info(f"Evaluation report plots saved to {plots_dir}")
    return saved_plots