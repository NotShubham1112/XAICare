"""
Cancer Prediction Engine for Clinical Use

Provides high-level prediction interface with:
- Single image prediction
- Batch prediction capabilities
- Confidence scoring and uncertainty estimation
- Integration with XAI for explainable predictions
"""

import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import yaml
import logging
from datetime import datetime

# Import project modules
from models.multi_cancer_model import create_multi_cancer_model, MultiCancerModel
from data_pipeline.preprocessing import MedicalImagePreprocessor
from xai.grad_cam import XAIInterpreter
from xai.explanations import ExplanationGenerator
from evaluation.metrics import calculate_clinical_metrics

logger = logging.getLogger(__name__)


class CancerPredictor:
    """
    Clinical cancer prediction engine.

    Provides production-ready prediction capabilities with
    comprehensive explainability for clinical decision support.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 cancer_types: Optional[List[str]] = None,
                 config_path: str = "config.yaml",
                 device: str = None):
        """
        Initialize cancer predictor.

        Args:
            model_path: Path to trained model checkpoint
            cancer_types: List of cancer types to support
            config_path: Path to configuration file
            device: Device for inference ('cuda', 'cpu', etc.)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize components
        self.preprocessor = MedicalImagePreprocessor(config_path)
        self.model = None
        self.xai_interpreter = None
        self.explanation_generator = ExplanationGenerator(config_path)

        # Load model
        if model_path:
            self.load_model(model_path, cancer_types)
        else:
            logger.warning("No model path provided. Call load_model() before prediction.")

        # Risk level thresholds
        self.risk_thresholds = self.config.get('evaluation', {}).get('risk_levels', {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        })

        logger.info(f"CancerPredictor initialized on device: {self.device}")

    def load_model(self, model_path: str, cancer_types: Optional[List[str]] = None):
        """
        Load trained model from checkpoint.

        Args:
            model_path: Path to model checkpoint
            cancer_types: Override cancer types if needed
        """
        # Load model
        self.model = MultiCancerModel.load_model(model_path, self.config_path)
        self.model.to(self.device)
        self.model.eval()

        # Override cancer types if specified
        if cancer_types:
            self.model.cancer_types = cancer_types
            self.model.cancer_classes = {
                ct: self.config['cancer_types'][self._get_cancer_index(ct)]['classes']
                for ct in cancer_types
            }

        # Initialize XAI interpreter
        self.xai_interpreter = XAIInterpreter(self.model, target_layer="layer4")

        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Supporting cancer types: {self.model.cancer_types}")

    def _get_cancer_index(self, cancer_type: str) -> int:
        """Get index of cancer type in config."""
        for i, cancer in enumerate(self.config['cancer_types']):
            if cancer['name'] == cancer_type:
                return i
        return 0

    def predict_single_image(self,
                           image_path: Union[str, Path],
                           cancer_type: str,
                           generate_explanation: bool = True,
                           save_visualizations: bool = False,
                           output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict cancer from single medical image.

        Args:
            image_path: Path to medical image
            cancer_type: Type of cancer to detect
            generate_explanation: Whether to generate XAI explanation
            save_visualizations: Whether to save visualization files
            output_dir: Directory to save visualizations

        Returns:
            Comprehensive prediction result dictionary
        """
        # Load and preprocess image
        image_tensor, original_image = self._load_and_preprocess_image(image_path, cancer_type)

        # Make prediction
        with torch.no_grad():
            prediction_result = self.model.predict_single_image(image_tensor, cancer_type)

        # Add clinical interpretation
        prediction_result = self._add_clinical_interpretation(prediction_result, cancer_type)

        # Generate explanation if requested
        if generate_explanation:
            explanation = self._generate_prediction_explanation(
                image_tensor, original_image, prediction_result, cancer_type
            )
            prediction_result['explanation'] = explanation

            # Generate visualizations
            if save_visualizations and output_dir:
                visualizations = self._save_prediction_visualizations(
                    image_tensor, original_image, explanation, cancer_type, output_dir
                )
                prediction_result['visualizations'] = visualizations

        # Add metadata
        prediction_result.update({
            'timestamp': datetime.now().isoformat(),
            'model_version': getattr(self.model, 'version', 'unknown'),
            'image_path': str(image_path),
            'processing_time_seconds': 0.0  # Could be measured
        })

        return prediction_result

    def _load_and_preprocess_image(self, image_path: Union[str, Path], cancer_type: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Load and preprocess medical image."""
        # Load image
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load with PIL
        image = Image.open(image_path)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy array
        original_image = np.array(image)

        # Get modality for cancer type
        modality = self._get_cancer_modality(cancer_type)

        # Preprocess image
        processed_image = self.preprocessor.preprocess_image(original_image, modality, augment=False)

        # Convert to tensor and add batch dimension
        if isinstance(processed_image, np.ndarray):
            image_tensor = torch.from_numpy(processed_image.transpose(2, 0, 1)).float().unsqueeze(0)
        else:
            image_tensor = processed_image.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        return image_tensor, original_image

    def _get_cancer_modality(self, cancer_type: str) -> str:
        """Get imaging modality for cancer type."""
        modality_map = {
            'lung': 'CT',
            'breast': 'mammogram',
            'brain': 'MRI',
            'skin': 'dermoscopy',
            'cervical': 'pap_smear',
            'colorectal': 'histopathology',
            'prostate': 'histopathology',
            'liver': 'CT'
        }
        return modality_map.get(cancer_type, 'CT')

    def _add_clinical_interpretation(self, prediction_result: Dict, cancer_type: str) -> Dict:
        """Add clinical interpretation to prediction results."""
        confidence = prediction_result.get('confidence', 0)
        predicted_class = prediction_result.get('prediction', 'unknown')

        # Determine risk level
        if confidence >= self.risk_thresholds['high']:
            risk_level = 'HIGH'
            clinical_significance = 'Strong evidence - immediate clinical attention recommended'
        elif confidence >= self.risk_thresholds['medium']:
            risk_level = 'MEDIUM'
            clinical_significance = 'Moderate evidence - further evaluation advised'
        elif confidence >= self.risk_thresholds['low']:
            risk_level = 'LOW'
            clinical_significance = 'Limited evidence - routine follow-up recommended'
        else:
            risk_level = 'VERY_LOW'
            clinical_significance = 'Minimal evidence - clinical correlation essential'

        # Add clinical fields
        prediction_result.update({
            'risk_level': risk_level,
            'clinical_significance': clinical_significance,
            'requires_followup': risk_level in ['HIGH', 'MEDIUM'],
            'urgency_level': 'urgent' if risk_level == 'HIGH' else 'routine',
            'cancer_type_specific': self._get_cancer_specific_info(cancer_type, predicted_class, confidence)
        })

        return prediction_result

    def _get_cancer_specific_info(self, cancer_type: str, predicted_class: str, confidence: float) -> Dict:
        """Get cancer-specific clinical information."""
        # This could be expanded with cancer-specific guidelines
        cancer_info = {
            'lung': {
                'followup_recommendation': 'CT chest within 1 month' if confidence > 0.8 else 'Routine screening',
                'biopsy_threshold': 0.7,
                'staging_relevance': 'Early detection critical for survival'
            },
            'breast': {
                'followup_recommendation': 'Diagnostic mammogram + ultrasound' if confidence > 0.8 else 'Routine screening',
                'biopsy_threshold': 0.6,
                'staging_relevance': 'Early detection improves 5-year survival by 20-30%'
            }
        }

        return cancer_info.get(cancer_type, {
            'followup_recommendation': 'Clinical correlation recommended',
            'biopsy_threshold': 0.8,
            'staging_relevance': 'Early detection beneficial'
        })

    def _generate_prediction_explanation(self,
                                       image_tensor: torch.Tensor,
                                       original_image: np.ndarray,
                                       prediction_result: Dict,
                                       cancer_type: str) -> Dict[str, Any]:
        """Generate comprehensive prediction explanation."""
        # Generate XAI visualizations
        xai_explanations = self.xai_interpreter.explain_prediction(
            image_tensor, original_image, cancer_type, method="gradcam"
        )

        # Get attention analysis
        attention_analysis = self.xai_interpreter.get_attention_regions(
            xai_explanations['heatmap'], threshold=0.5
        )

        # Generate textual explanation
        textual_explanation = self.explanation_generator.generate_explanation(
            prediction_result, attention_analysis, cancer_type
        )

        explanation = {
            'textual_explanation': textual_explanation,
            'attention_analysis': attention_analysis,
            'visual_explanations': {
                'heatmap_available': 'heatmap' in xai_explanations,
                'overlay_available': 'overlay' in xai_explanations,
                'saliency_available': 'saliency' in xai_explanations
            },
            'confidence_interpretation': self._interpret_confidence(prediction_result['confidence']),
            'clinical_recommendations': self._get_explanation_recommendations(
                prediction_result, attention_analysis, cancer_type
            )
        }

        return explanation

    def _interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence score in clinical context."""
        if confidence >= 0.9:
            return "Very high confidence - results highly reliable"
        elif confidence >= 0.8:
            return "High confidence - results generally reliable"
        elif confidence >= 0.7:
            return "Moderate confidence - results should be correlated with clinical findings"
        elif confidence >= 0.6:
            return "Low confidence - additional testing recommended"
        else:
            return "Very low confidence - expert review essential"

    def _get_explanation_recommendations(self,
                                       prediction_result: Dict,
                                       attention_analysis: Dict,
                                       cancer_type: str) -> List[str]:
        """Generate explanation-based clinical recommendations."""
        recommendations = []
        confidence = prediction_result.get('confidence', 0)
        risk_level = prediction_result.get('risk_level', 'LOW')

        # Base recommendations
        if risk_level == 'HIGH':
            recommendations.append("Immediate specialist consultation recommended")
            if cancer_type in ['lung', 'breast']:
                recommendations.append("Consider expedited biopsy or further imaging within 1-2 weeks")

        elif risk_level == 'MEDIUM':
            recommendations.append("Further clinical evaluation recommended")
            recommendations.append("Correlation with patient history and additional imaging advised")

        # Attention-based recommendations
        num_regions = attention_analysis.get('num_regions', 0)
        if num_regions > 1:
            recommendations.append("Multiple suspicious regions identified - comprehensive evaluation needed")

        # Cancer-specific recommendations
        if cancer_type == 'lung' and confidence > 0.8:
            recommendations.append("Consider low-dose CT follow-up per Fleischner guidelines")
        elif cancer_type == 'breast' and confidence > 0.7:
            recommendations.append("BI-RADS category assessment recommended")

        return recommendations

    def _save_prediction_visualizations(self,
                                      image_tensor: torch.Tensor,
                                      original_image: np.ndarray,
                                      explanation: Dict,
                                      cancer_type: str,
                                      output_dir: str) -> Dict[str, str]:
        """Save prediction visualizations to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{cancer_type}_prediction_{timestamp}"

        saved_files = {}

        # Save original image
        original_path = output_dir / f"{base_filename}_original.png"
        cv2.imwrite(str(original_path), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        saved_files['original_image'] = str(original_path)

        # Generate and save XAI visualizations
        xai_explanations = self.xai_interpreter.explain_prediction(
            image_tensor, original_image, cancer_type, method="gradcam"
        )

        # Save heatmap
        if 'heatmap' in xai_explanations:
            heatmap_path = output_dir / f"{base_filename}_heatmap.png"
            cv2.imwrite(str(heatmap_path), xai_explanations['heatmap'] * 255)
            saved_files['heatmap'] = str(heatmap_path)

        # Save overlay
        if 'overlay' in xai_explanations:
            overlay_path = output_dir / f"{base_filename}_overlay.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(xai_explanations['overlay'], cv2.COLOR_RGB2BGR))
            saved_files['overlay'] = str(overlay_path)

        logger.info(f"Prediction visualizations saved to {output_dir}")
        return saved_files

    def predict_batch(self,
                     image_paths: List[Union[str, Path]],
                     cancer_type: str,
                     batch_size: int = 16,
                     generate_explanations: bool = False) -> List[Dict[str, Any]]:
        """
        Predict cancer from batch of medical images.

        Args:
            image_paths: List of paths to medical images
            cancer_type: Type of cancer to detect
            batch_size: Batch size for processing
            generate_explanations: Whether to generate explanations

        Returns:
            List of prediction results
        """
        results = []

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_results = []

            for image_path in batch_paths:
                try:
                    result = self.predict_single_image(
                        image_path, cancer_type,
                        generate_explanation=generate_explanations,
                        save_visualizations=False
                    )
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    batch_results.append({
                        'error': str(e),
                        'image_path': str(image_path),
                        'cancer_type': cancer_type
                    })

            results.extend(batch_results)

        logger.info(f"Processed {len(results)} images in batch mode")
        return results

    def get_prediction_statistics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from batch predictions."""
        if not predictions:
            return {}

        # Extract valid predictions
        valid_predictions = [p for p in predictions if 'error' not in p]

        if not valid_predictions:
            return {'error': 'No valid predictions found'}

        # Calculate statistics
        confidences = [p.get('confidence', 0) for p in valid_predictions]
        risk_levels = [p.get('risk_level', 'UNKNOWN') for p in valid_predictions]
        predictions_class = [p.get('prediction', 'unknown') for p in valid_predictions]

        # Risk level distribution
        risk_distribution = {}
        for level in ['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH']:
            risk_distribution[level] = risk_levels.count(level)

        # Class distribution
        class_distribution = {}
        for pred_class in set(predictions_class):
            class_distribution[pred_class] = predictions_class.count(pred_class)

        stats = {
            'total_predictions': len(predictions),
            'valid_predictions': len(valid_predictions),
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'risk_distribution': risk_distribution,
            'class_distribution': class_distribution,
            'high_risk_count': risk_distribution.get('HIGH', 0),
            'high_risk_percentage': risk_distribution.get('HIGH', 0) / len(valid_predictions) * 100
        }

        return stats

    def validate_prediction_quality(self,
                                  validation_images: List[Union[str, Path]],
                                  validation_labels: List[int],
                                  cancer_type: str) -> Dict[str, float]:
        """
        Validate prediction quality on held-out data.

        Args:
            validation_images: List of validation image paths
            validation_labels: Corresponding ground truth labels
            cancer_type: Cancer type being validated

        Returns:
            Validation metrics
        """
        predictions = self.predict_batch(validation_images, cancer_type, generate_explanations=False)

        # Extract predictions and probabilities
        y_pred = []
        y_prob = []
        valid_indices = []

        for i, pred in enumerate(predictions):
            if 'error' not in pred:
                y_pred.append(1 if pred.get('prediction') == 'malignant' else 0)
                y_prob.append(pred.get('confidence', 0))
                valid_indices.append(i)

        if not y_pred:
            return {'error': 'No valid predictions for validation'}

        # Filter validation labels
        y_true_filtered = [validation_labels[i] for i in valid_indices]

        # Calculate clinical metrics
        clinical_metrics = calculate_clinical_metrics(
            np.array(y_true_filtered),
            np.array(y_pred),
            np.array(y_prob)
        )

        clinical_metrics.update({
            'validation_samples': len(valid_indices),
            'cancer_type': cancer_type,
            'timestamp': datetime.now().isoformat()
        })

        return clinical_metrics