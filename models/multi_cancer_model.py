"""
Multi-Cancer Detection Model

Combines shared backbone with cancer-specific classification heads
for unified multi-cancer early detection.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
import yaml
import logging
from .backbone import get_backbone, unfreeze_backbone_layers
from .heads import MultiCancerHeads, SegmentationHead

logger = logging.getLogger(__name__)


class MultiCancerModel(nn.Module):
    """
    Multi-cancer early detection model with shared backbone.

    Architecture:
    - Shared feature extractor (ResNet, EfficientNet, etc.)
    - Cancer-specific classification heads
    - Optional segmentation head for tumor localization
    """

    def __init__(self,
                 cancer_configs: List[Dict],
                 backbone_name: str = "resnet50",
                 pretrained: bool = True,
                 freeze_backbone: bool = True,
                 use_segmentation: bool = False,
                 config_path: str = "config.yaml"):
        """
        Initialize multi-cancer model.

        Args:
            cancer_configs: List of cancer type configurations
            backbone_name: Name of backbone architecture
            pretrained: Whether to use pre-trained weights
            freeze_backbone: Whether to freeze backbone parameters
            use_segmentation: Whether to include segmentation head
            config_path: Path to configuration file
        """
        super().__init__()

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.cancer_configs = cancer_configs
        self.backbone_name = backbone_name
        self.use_segmentation = use_segmentation

        # Create shared backbone
        self.backbone, self.feature_dim = get_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            config_path=config_path
        )

        # Create cancer-specific classification heads
        self.classification_heads = MultiCancerHeads(
            input_dim=self.feature_dim,
            cancer_configs=cancer_configs,
            config_path=config_path
        )

        # Optional segmentation head
        self.segmentation_head = None
        if use_segmentation:
            self.segmentation_head = SegmentationHead(
                input_dim=self.feature_dim,
                num_classes=2,  # Background and tumor
                config_path=config_path
            )

        # Store cancer type information
        self.cancer_names = [config['name'] for config in cancer_configs]
        self.cancer_classes = {config['name']: config['classes'] for config in cancer_configs}

        logger.info(f"Initialized MultiCancerModel with backbone: {backbone_name}")
        logger.info(f"Cancer types: {self.cancer_names}")
        logger.info(f"Feature dimension: {self.feature_dim}")

    def forward(self,
                x: torch.Tensor,
                cancer_type: Optional[str] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input image batch
            cancer_type: Specific cancer type to classify (None for all)
            return_features: Whether to return backbone features

        Returns:
            Dictionary containing predictions and optional features
        """
        # Extract features using shared backbone
        features = self.backbone(x)

        # Store features for potential use in explainability
        if return_features:
            # Reshape features back to spatial dimensions for Grad-CAM
            # This assumes the backbone outputs flattened features
            batch_size = x.shape[0]
            spatial_features = features.view(batch_size, self.feature_dim, 1, 1)

        # Get classification outputs
        classification_outputs = self.classification_heads(features, cancer_type)

        # Prepare final outputs
        outputs = classification_outputs

        # Add features if requested
        if return_features:
            outputs['features'] = features
            outputs['spatial_features'] = spatial_features

        # Add segmentation if enabled
        if self.segmentation_head is not None and return_features:
            # For segmentation, we need spatial feature maps from backbone
            # This requires modifying backbone to return intermediate features
            segmentation_logits = self.segmentation_head(spatial_features)
            outputs['segmentation_logits'] = segmentation_logits

        return outputs

    def get_classification_head(self, cancer_type: str) -> nn.Module:
        """Get classification head for specific cancer type."""
        return self.classification_heads.get_head(cancer_type)

    def get_classification_weights(self, cancer_type: str) -> torch.Tensor:
        """Get classification weights for Grad-CAM."""
        return self.classification_heads.get_classification_weights(cancer_type)

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Froze backbone parameters")

    def unfreeze_backbone_layers(self, layers_to_unfreeze: List[int]):
        """Unfreeze specific backbone layers for fine-tuning."""
        self.backbone = unfreeze_backbone_layers(
            self.backbone,
            self.backbone_name,
            layers_to_unfreeze
        )

    def freeze_classification_heads(self, cancer_types: Optional[List[str]] = None):
        """Freeze classification heads for specified cancer types."""
        if cancer_types is None:
            cancer_types = self.cancer_names

        for cancer_type in cancer_types:
            self.classification_heads.freeze_head(cancer_type)

    def unfreeze_classification_heads(self, cancer_types: Optional[List[str]] = None):
        """Unfreeze classification heads for specified cancer types."""
        if cancer_types is None:
            cancer_types = self.cancer_names

        for cancer_type in cancer_types:
            self.classification_heads.unfreeze_head(cancer_type)

    def get_trainable_parameters(self) -> List[torch.Tensor]:
        """Get all trainable parameters."""
        return [param for param in self.parameters() if param.requires_grad]

    def get_parameter_groups(self, lr_backbone: float, lr_heads: float) -> List[Dict]:
        """
        Get parameter groups for differential learning rates.

        Args:
            lr_backbone: Learning rate for backbone
            lr_heads: Learning rate for classification heads

        Returns:
            List of parameter groups for optimizer
        """
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.classification_heads.parameters())

        # Add segmentation head if present
        if self.segmentation_head is not None:
            head_params.extend(list(self.segmentation_head.parameters()))

        param_groups = [
            {'params': backbone_params, 'lr': lr_backbone, 'name': 'backbone'},
            {'params': head_params, 'lr': lr_heads, 'name': 'heads'}
        ]

        return param_groups

    def save_model(self, filepath: str, metadata: Optional[Dict] = None):
        """Save model state and metadata."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'backbone_name': self.backbone_name,
            'cancer_configs': self.cancer_configs,
            'feature_dim': self.feature_dim,
            'use_segmentation': self.use_segmentation,
            'metadata': metadata or {}
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str, config_path: str = "config.yaml") -> 'MultiCancerModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')

        # Recreate model
        model = cls(
            cancer_configs=checkpoint['cancer_configs'],
            backbone_name=checkpoint['backbone_name'],
            use_segmentation=checkpoint.get('use_segmentation', False),
            config_path=config_path
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Model loaded from {filepath}")
        return model

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'backbone': self.backbone_name,
            'cancer_types': self.cancer_names,
            'feature_dimension': self.feature_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'uses_segmentation': self.segmentation_head is not None,
            'cancer_classes': self.cancer_classes
        }

        return info

    def predict_single_image(self,
                           image: torch.Tensor,
                           cancer_type: str) -> Dict[str, Any]:
        """
        Predict on single image for specific cancer type.

        Args:
            image: Single image tensor
            cancer_type: Cancer type to classify

        Returns:
            Prediction results dictionary
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension if needed
            if image.dim() == 3:
                image = image.unsqueeze(0)

            # Forward pass
            outputs = self.forward(image, cancer_type=cancer_type)

            # Convert to CPU and numpy
            prediction = outputs['prediction'].cpu().item()
            probabilities = outputs['probabilities'].cpu().numpy()[0]
            uncertainty = outputs['uncertainty'].cpu().item()

            # Get class names
            classes = self.cancer_classes[cancer_type]

            result = {
                'cancer_type': cancer_type,
                'prediction': classes[prediction],
                'confidence': float(probabilities[prediction]),
                'probabilities': {classes[i]: float(prob) for i, prob in enumerate(probabilities)},
                'uncertainty': uncertainty,
                'classes': classes
            }

            return result

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.backbone, 'gradient_checkpointing_enable'):
            self.backbone.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")


def create_multi_cancer_model(cancer_types: List[str],
                             backbone_name: str = "resnet50",
                             config_path: str = "config.yaml") -> MultiCancerModel:
    """
    Factory function to create multi-cancer model.

    Args:
        cancer_types: List of cancer types to include
        backbone_name: Backbone architecture name
        config_path: Path to configuration file

    Returns:
        Configured MultiCancerModel
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get cancer configurations
    cancer_configs = []
    for cancer_config in config['cancer_types']:
        if cancer_config['name'] in cancer_types:
            cancer_configs.append(cancer_config)

    if not cancer_configs:
        raise ValueError(f"No valid cancer types found in: {cancer_types}")

    # Create model
    model = MultiCancerModel(
        cancer_configs=cancer_configs,
        backbone_name=backbone_name,
        config_path=config_path
    )

    return model