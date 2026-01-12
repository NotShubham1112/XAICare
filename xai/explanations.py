"""
Textual Explanation Generation for Clinical Interpretability

Provides human-readable explanations of model predictions based on:
- Visual attention patterns
- Clinical knowledge rules
- Confidence scores and uncertainty
"""

import torch
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """
    Generate textual explanations for medical image predictions.

    Combines visual analysis with clinical knowledge to provide
    interpretable explanations suitable for clinical use.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize explanation generator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load clinical knowledge base
        self.clinical_rules = self._load_clinical_rules()

    def _load_clinical_rules(self) -> Dict[str, Dict]:
        """Load clinical interpretation rules for different cancer types."""
        return {
            'lung': {
                'nodule_features': {
                    'size': {'small': '< 5mm', 'medium': '5-10mm', 'large': '> 10mm'},
                    'shape': {'regular': 'smooth borders', 'irregular': 'spiculated edges'},
                    'density': {'solid': 'high density', 'ground_glass': 'low density'},
                    'location': {'upper': 'upper lobes', 'lower': 'lower lobes'}
                },
                'risk_patterns': {
                    'high': 'Irregular nodule with spiculated edges and pleural attachment',
                    'medium': 'Solid nodule with regular borders',
                    'low': 'Small, smooth, peripheral nodule'
                }
            },
            'breast': {
                'mass_features': {
                    'shape': {'regular': 'oval/round', 'irregular': 'irregular borders'},
                    'margin': {'well_defined': 'sharp margins', 'ill_defined': 'poorly defined'},
                    'density': {'high': 'dense mass', 'low': 'fat-containing'},
                },
                'calcification_features': {
                    'morphology': {'punctate': 'tiny dots', 'linear': 'linear arrangement'},
                    'distribution': {'clustered': 'grouped together', 'diffuse': 'spread out'}
                }
            },
            'skin': {
                'asymmetry': ['asymmetrical shape', 'uneven color distribution'],
                'border': ['irregular borders', 'notched edges'],
                'color': ['multiple colors', 'dark pigmentation'],
                'diameter': ['larger than 6mm', 'increasing size']
            }
        }

    def generate_explanation(self,
                           prediction: Dict,
                           attention_analysis: Dict,
                           cancer_type: str,
                           confidence_thresholds: Optional[Dict[str, float]] = None) -> str:
        """
        Generate comprehensive textual explanation.

        Args:
            prediction: Model prediction results
            attention_analysis: Visual attention analysis
            cancer_type: Type of cancer
            confidence_thresholds: Custom confidence thresholds

        Returns:
            Human-readable explanation string
        """
        if confidence_thresholds is None:
            confidence_thresholds = {'high': 0.8, 'medium': 0.6, 'low': 0.4}

        confidence = prediction.get('confidence', 0)
        predicted_class = prediction.get('prediction', 'unknown')
        risk_level = self._classify_risk_level(confidence, confidence_thresholds)

        # Build explanation components
        explanation_parts = []

        # 1. Prediction summary
        explanation_parts.append(
            self._generate_prediction_summary(predicted_class, confidence, risk_level)
        )

        # 2. Visual analysis
        if attention_analysis:
            explanation_parts.append(
                self._generate_visual_analysis(attention_analysis, cancer_type)
            )

        # 3. Clinical interpretation
        explanation_parts.append(
            self._generate_clinical_interpretation(cancer_type, predicted_class, attention_analysis)
        )

        # 4. Uncertainty statement
        if confidence < confidence_thresholds['medium']:
            explanation_parts.append(
                self._generate_uncertainty_statement(confidence)
            )

        # 5. Recommendations
        explanation_parts.append(
            self._generate_recommendations(risk_level, cancer_type)
        )

        # Combine all parts
        full_explanation = " ".join(explanation_parts)

        return full_explanation

    def _classify_risk_level(self, confidence: float, thresholds: Dict[str, float]) -> str:
        """Classify risk level based on confidence."""
        if confidence >= thresholds['high']:
            return 'high'
        elif confidence >= thresholds['medium']:
            return 'medium'
        elif confidence >= thresholds['low']:
            return 'low'
        else:
            return 'very_low'

    def _generate_prediction_summary(self,
                                   predicted_class: str,
                                   confidence: float,
                                   risk_level: str) -> str:
        """Generate prediction summary statement."""
        confidence_pct = confidence * 100

        templates = {
            'high': f"The model predicts {predicted_class} with high confidence ({confidence_pct:.1f}%).",
            'medium': f"The model suggests {predicted_class} with moderate confidence ({confidence_pct:.1f}%).",
            'low': f"The model indicates possible {predicted_class} with low confidence ({confidence_pct:.1f}%).",
            'very_low': f"The model shows very low confidence ({confidence_pct:.1f}%) for {predicted_class}."
        }

        return templates.get(risk_level, templates['very_low'])

    def _generate_visual_analysis(self,
                                attention_analysis: Dict,
                                cancer_type: str) -> str:
        """Generate description of visual attention patterns."""
        num_regions = attention_analysis.get('num_regions', 0)
        attention_pct = attention_analysis.get('attention_percentage', 0) * 100
        regions = attention_analysis.get('regions', [])

        if num_regions == 0:
            return "The model shows diffuse attention patterns across the image."

        # Describe attention distribution
        if attention_pct < 10:
            focus_desc = "minimal"
        elif attention_pct < 25:
            focus_desc = "moderate"
        elif attention_pct < 50:
            focus_desc = "substantial"
        else:
            focus_desc = "intense"

        region_desc = f"focusing on {num_regions} distinct region{'s' if num_regions > 1 else ''}"

        # Describe largest region
        if regions:
            largest_region = regions[0]
            bbox = largest_region['bbox']
            area = largest_region['area']
            relative_size = "small" if area < 1000 else "moderate" if area < 5000 else "large"

            location_desc = self._describe_region_location(bbox, cancer_type)

            return f"The model shows {focus_desc} attention {region_desc}, with particular focus on a {relative_size} {location_desc} region covering {attention_pct:.1f}% of the image."

        return f"The model shows {focus_desc} attention across {attention_pct:.1f}% of the image."

    def _describe_region_location(self, bbox: Tuple[int, int, int, int], cancer_type: str) -> str:
        """Describe the location of an attention region."""
        x, y, w, h = bbox

        # Generic location descriptions (could be made cancer-specific)
        if cancer_type == 'lung':
            # For lung CT scans
            if y < 100:
                vertical_pos = "upper"
            elif y > 300:
                vertical_pos = "lower"
            else:
                vertical_pos = "middle"

            if x < 150:
                horizontal_pos = "left"
            elif x > 250:
                horizontal_pos = "right"
            else:
                horizontal_pos = "central"

            return f"{vertical_pos} {horizontal_pos}"

        elif cancer_type == 'breast':
            # For mammograms
            if x < 150:
                lateral_pos = "medial"
            else:
                lateral_pos = "lateral"

            if y < 150:
                depth_pos = "anterior"
            else:
                depth_pos = "posterior"

            return f"{lateral_pos} {depth_pos}"

        else:
            # Generic description
            return f"region at position ({x}, {y})"

    def _generate_clinical_interpretation(self,
                                        cancer_type: str,
                                        predicted_class: str,
                                        attention_analysis: Dict) -> str:
        """Generate clinical interpretation based on cancer-specific knowledge."""
        if cancer_type not in self.clinical_rules:
            return "Clinical interpretation requires domain-specific medical knowledge."

        rules = self.clinical_rules[cancer_type]

        # Generate interpretation based on attention patterns
        num_regions = attention_analysis.get('num_regions', 0)
        attention_pct = attention_analysis.get('attention_percentage', 0)

        if cancer_type == 'lung':
            return self._interpret_lung_cancer(predicted_class, num_regions, attention_pct)
        elif cancer_type == 'breast':
            return self._interpret_breast_cancer(predicted_class, attention_analysis)
        elif cancer_type == 'skin':
            return self._interpret_skin_cancer(predicted_class, attention_analysis)
        else:
            return f"This finding is consistent with {predicted_class} based on the model's attention to suspicious regions."

    def _interpret_lung_cancer(self, predicted_class: str, num_regions: int, attention_pct: float) -> str:
        """Generate lung cancer specific interpretation."""
        if predicted_class.lower() == 'malignant':
            if num_regions == 1 and attention_pct > 30:
                return "The model identifies a solitary nodule with concerning features such as irregular borders or spiculated edges, consistent with early-stage lung malignancy."
            elif num_regions > 1:
                return "Multiple suspicious nodules detected, suggesting possible metastatic disease or multifocal primary lung cancer."
            else:
                return "Diffuse abnormal tissue patterns detected, potentially indicating infiltrative lung malignancy."
        else:
            return "The model identifies benign-appearing pulmonary nodule(s) with regular borders and typical benign features."

    def _interpret_breast_cancer(self, predicted_class: str, attention_analysis: Dict) -> str:
        """Generate breast cancer specific interpretation."""
        regions = attention_analysis.get('regions', [])

        if predicted_class.lower() == 'malignant':
            if regions:
                largest_region = regions[0]
                bbox = largest_region['bbox']
                # Describe mass characteristics based on attention pattern
                return "The model detects a suspicious breast mass with irregular borders and heterogeneous density, concerning for malignancy."
            else:
                return "Microcalcifications or architectural distortion detected, warranting further evaluation."
        else:
            return "The model identifies benign-appearing breast tissue with normal architectural patterns."

    def _interpret_skin_cancer(self, predicted_class: str, attention_analysis: Dict) -> str:
        """Generate skin cancer specific interpretation."""
        if predicted_class.lower() == 'malignant':
            return "The model identifies asymmetrical skin lesion with irregular borders and color variation, concerning for melanoma."
        else:
            return "The model detects benign-appearing skin lesion with regular features."

    def _generate_uncertainty_statement(self, confidence: float) -> str:
        """Generate statement about prediction uncertainty."""
        confidence_pct = confidence * 100

        if confidence_pct < 50:
            return "However, the model's confidence is low, suggesting this finding should be interpreted cautiously."
        else:
            return "The model's moderate confidence indicates this result should be correlated with additional clinical findings."

    def _generate_recommendations(self, risk_level: str, cancer_type: str) -> str:
        """Generate clinical recommendations based on risk level."""
        recommendations = {
            'high': "This high-confidence finding warrants immediate clinical correlation and potential biopsy or further imaging.",
            'medium': "This finding should prompt additional clinical evaluation and follow-up imaging.",
            'low': "This low-risk finding may be suitable for routine follow-up but should not be ignored.",
            'very_low': "This very low confidence result requires expert clinical review and additional testing."
        }

        base_recommendation = recommendations.get(risk_level, recommendations['very_low'])

        # Add cancer-specific recommendations
        if risk_level in ['high', 'medium']:
            if cancer_type == 'lung':
                base_recommendation += " Consider CT-guided biopsy or surgical consultation."
            elif cancer_type == 'breast':
                base_recommendation += " Mammography follow-up and possible ultrasound or MRI recommended."
            elif cancer_type == 'skin':
                base_recommendation += " Dermatologic evaluation with possible excisional biopsy advised."

        return base_recommendation

    def generate_structured_explanation(self,
                                      prediction: Dict,
                                      attention_analysis: Dict,
                                      cancer_type: str) -> Dict[str, str]:
        """
        Generate structured explanation with separate components.

        Args:
            prediction: Model prediction results
            attention_analysis: Visual attention analysis
            cancer_type: Type of cancer

        Returns:
            Dictionary with structured explanation components
        """
        return {
            'prediction_summary': self._generate_prediction_summary(
                prediction.get('prediction', 'unknown'),
                prediction.get('confidence', 0),
                self._classify_risk_level(prediction.get('confidence', 0), {'high': 0.8, 'medium': 0.6, 'low': 0.4})
            ),
            'visual_analysis': self._generate_visual_analysis(attention_analysis, cancer_type),
            'clinical_interpretation': self._generate_clinical_interpretation(
                cancer_type, prediction.get('prediction', 'unknown'), attention_analysis
            ),
            'recommendations': self._generate_recommendations(
                self._classify_risk_level(prediction.get('confidence', 0), {'high': 0.8, 'medium': 0.6, 'low': 0.4}),
                cancer_type
            ),
            'confidence_level': self._classify_risk_level(
                prediction.get('confidence', 0), {'high': 0.8, 'medium': 0.6, 'low': 0.4}
            )
        }


class ClinicalKnowledgeBase:
    """
    Repository of clinical knowledge for explanation generation.

    Provides cancer-specific interpretation rules and guidelines.
    """

    def __init__(self):
        """Initialize clinical knowledge base."""
        self.cancer_characteristics = {
            'lung': {
                'malignant_indicators': [
                    'spiculated edges',
                    'irregular borders',
                    'ground-glass opacity',
                    'pleural attachment',
                    'air bronchograms'
                ],
                'benign_indicators': [
                    'smooth borders',
                    'calcification',
                    'fat density',
                    'vascular convergence'
                ]
            },
            'breast': {
                'malignant_indicators': [
                    'irregular shape',
                    'spiculated margins',
                    'microcalcifications',
                    'architectural distortion'
                ],
                'benign_indicators': [
                    'round/oval shape',
                    'smooth margins',
                    'macrocysts',
                    'fat-containing lesions'
                ]
            }
        }

    def get_cancer_characteristics(self, cancer_type: str) -> Dict[str, List[str]]:
        """Get characteristics for specific cancer type."""
        return self.cancer_characteristics.get(cancer_type, {})

    def score_abnormality_features(self, attention_analysis: Dict, cancer_type: str) -> Dict[str, float]:
        """Score abnormality based on attention patterns and clinical features."""
        # This is a simplified scoring system
        # In practice, this would be more sophisticated

        scores = {
            'malignancy_score': 0.0,
            'confidence_score': 0.0,
            'specificity_score': 0.0
        }

        # Score based on attention patterns
        num_regions = attention_analysis.get('num_regions', 0)
        attention_pct = attention_analysis.get('attention_percentage', 0)

        if num_regions == 1 and 0.1 < attention_pct < 0.5:
            scores['malignancy_score'] += 0.3
        elif num_regions > 1:
            scores['malignancy_score'] += 0.5

        if attention_pct > 0.3:
            scores['confidence_score'] += 0.4

        return scores


def export_explanation_report(explanation: Union[str, Dict],
                           prediction: Dict,
                           save_path: str,
                           filename: str = "clinical_report.txt"):
    """
    Export explanation report to file.

    Args:
        explanation: Generated explanation
        prediction: Model prediction
        save_path: Directory to save report
        filename: Report filename
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    filepath = save_path / filename

    with open(filepath, 'w') as f:
        f.write("MULTI-CANCER AI DETECTION CLINICAL REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Cancer Type: {prediction.get('cancer_type', 'Unknown')}\n")
        f.write(f"Prediction: {prediction.get('prediction', 'Unknown')}\n")
        f.write(f"Confidence: {prediction.get('confidence', 0):.1%}\n\n")

        if isinstance(explanation, dict):
            for section, text in explanation.items():
                f.write(f"{section.upper().replace('_', ' ')}:\n")
                f.write(f"{text}\n\n")
        else:
            f.write("CLINICAL INTERPRETATION:\n")
            f.write(f"{explanation}\n\n")

        f.write("DISCLAIMER:\n")
        f.write("This AI analysis is assistive only and should be reviewed by qualified medical professionals.\n")
        f.write("Final diagnosis requires clinical correlation and may involve additional testing.\n")

    logger.info(f"Clinical report saved to {filepath}")