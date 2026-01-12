"""
Shared Backbone Architectures for Multi-Cancer Detection

Provides pre-trained feature extractors with transfer learning support:
- ResNet (18, 34, 50, 101, 152)
- EfficientNet (B0-B7)
- DenseNet (121, 161, 169, 201)
"""

import torch
import torch.nn as nn
from torchvision import models
import timm
from typing import Optional, Dict, Any, Tuple
import yaml
import logging

logger = logging.getLogger(__name__)


def get_backbone(backbone_name: str = "resnet50",
                pretrained: bool = True,
                freeze_backbone: bool = True,
                config_path: str = "config.yaml") -> Tuple[nn.Module, int]:
    """
    Get pre-trained backbone with feature extraction capabilities.

    Args:
        backbone_name: Name of backbone architecture
        pretrained: Whether to use pre-trained weights
        freeze_backbone: Whether to freeze backbone parameters
        config_path: Path to configuration file

    Returns:
        Tuple of (backbone_model, feature_dim)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override with function parameters
    if 'backbone' in config['model']:
        if backbone_name == "resnet50":  # Default, use config
            backbone_name = config['model']['backbone']

    logger.info(f"Initializing backbone: {backbone_name}")

    if backbone_name.startswith('resnet'):
        model, feature_dim = _get_resnet_backbone(backbone_name, pretrained)
    elif backbone_name.startswith('efficientnet'):
        model, feature_dim = _get_efficientnet_backbone(backbone_name, pretrained)
    elif backbone_name.startswith('densenet'):
        model, feature_dim = _get_densenet_backbone(backbone_name, pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Backbone parameters frozen for transfer learning")

    return model, feature_dim


def _get_resnet_backbone(model_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    """Get ResNet backbone."""
    model_dict = {
        'resnet18': (models.resnet18, 512),
        'resnet34': (models.resnet34, 512),
        'resnet50': (models.resnet50, 2048),
        'resnet101': (models.resnet101, 2048),
        'resnet152': (models.resnet152, 2048)
    }

    if model_name not in model_dict:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")

    model_fn, feature_dim = model_dict[model_name]

    # Load pre-trained model
    model = model_fn(pretrained=pretrained)

    # Remove the final classification layer
    model = nn.Sequential(*list(model.children())[:-2])  # Remove avgpool and fc

    # Add adaptive average pooling to get fixed-size features
    model = nn.Sequential(
        model,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )

    return model, feature_dim


def _get_efficientnet_backbone(model_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    """Get EfficientNet backbone."""
    # Map model names to timm names
    model_name_map = {
        'efficientnet_b0': 'efficientnet_b0',
        'efficientnet_b1': 'efficientnet_b1',
        'efficientnet_b2': 'efficientnet_b2',
        'efficientnet_b3': 'efficientnet_b3',
        'efficientnet_b4': 'efficientnet_b4',
        'efficientnet_b5': 'efficientnet_b5',
        'efficientnet_b6': 'efficientnet_b6',
        'efficientnet_b7': 'efficientnet_b7'
    }

    if model_name not in model_name_map:
        raise ValueError(f"Unsupported EfficientNet variant: {model_name}")

    timm_name = model_name_map[model_name]

    # Load model using timm
    model = timm.create_model(timm_name, pretrained=pretrained, features_only=True)

    # Get feature dimension from the last feature layer
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        features = model(dummy_input)
        feature_dim = features[-1].shape[1]

    # Use only the last feature layer and add pooling
    class EfficientNetFeatures(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            features = self.base_model(x)
            # Use the last feature map
            x = features[-1]
            # Global average pooling
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = x.flatten(1)
            return x

    model = EfficientNetFeatures(model)

    return model, feature_dim


def _get_densenet_backbone(model_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    """Get DenseNet backbone."""
    model_dict = {
        'densenet121': (models.densenet121, 1024),
        'densenet161': (models.densenet161, 2208),
        'densenet169': (models.densenet169, 1664),
        'densenet201': (models.densenet201, 1920)
    }

    if model_name not in model_dict:
        raise ValueError(f"Unsupported DenseNet variant: {model_name}")

    model_fn, feature_dim = model_dict[model_name]

    # Load pre-trained model
    model = model_fn(pretrained=pretrained)

    # Remove the classifier
    model = model.features

    # Add adaptive average pooling
    model = nn.Sequential(
        model,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )

    return model, feature_dim


def unfreeze_backbone_layers(model: nn.Module,
                           backbone_name: str,
                           layers_to_unfreeze: list) -> nn.Module:
    """
    Progressively unfreeze backbone layers for fine-tuning.

    Args:
        model: Backbone model
        backbone_name: Name of backbone architecture
        layers_to_unfreeze: List of layer indices to unfreeze

    Returns:
        Model with selectively unfrozen layers
    """
    if backbone_name.startswith('resnet'):
        _unfreeze_resnet_layers(model, layers_to_unfreeze)
    elif backbone_name.startswith('efficientnet'):
        _unfreeze_efficientnet_layers(model, layers_to_unfreeze)
    elif backbone_name.startswith('densenet'):
        _unfreeze_densenet_layers(model, layers_to_unfreeze)

    return model


def _unfreeze_resnet_layers(model: nn.Module, layers_to_unfreeze: list):
    """Unfreeze specific ResNet layers."""
    layer_names = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']

    for layer_idx in layers_to_unfreeze:
        if layer_idx < len(layer_names):
            layer_name = layer_names[layer_idx]
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True
                logger.info(f"Unfroze ResNet layer: {layer_name}")


def _unfreeze_efficientnet_layers(model: nn.Module, layers_to_unfreeze: list):
    """Unfreeze specific EfficientNet layers."""
    # EfficientNet layers are organized differently
    # This is a simplified version - would need more sophisticated unfreezing
    if hasattr(model, 'base_model'):
        base_model = model.base_model
        # Unfreeze blocks progressively
        if hasattr(base_model, 'blocks'):
            blocks = base_model.blocks
            for layer_idx in layers_to_unfreeze:
                if layer_idx < len(blocks):
                    for param in blocks[layer_idx].parameters():
                        param.requires_grad = True
                    logger.info(f"Unfroze EfficientNet block: {layer_idx}")


def _unfreeze_densenet_layers(model: nn.Module, layers_to_unfreeze: list):
    """Unfreeze specific DenseNet layers."""
    # DenseNet has denseblocks
    denseblock_names = ['denseblock1', 'denseblock2', 'denseblock3', 'denseblock4']

    for layer_idx in layers_to_unfreeze:
        if layer_idx < len(denseblock_names):
            block_name = denseblock_names[layer_idx]
            if hasattr(model, block_name):
                block = getattr(model, block_name)
                for param in block.parameters():
                    param.requires_grad = True
                logger.info(f"Unfroze DenseNet block: {block_name}")


def get_feature_extractor_info(backbone_name: str) -> Dict[str, Any]:
    """
    Get information about backbone feature extractor.

    Args:
        backbone_name: Name of backbone architecture

    Returns:
        Dictionary with backbone information
    """
    info = {
        'name': backbone_name,
        'supports_feature_extraction': True,
        'recommended_input_size': (224, 224),
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }

    if backbone_name.startswith('resnet'):
        info.update({
            'architecture': 'CNN',
            'feature_type': 'global_average_pooled'
        })
    elif backbone_name.startswith('efficientnet'):
        info.update({
            'architecture': 'CNN',
            'feature_type': 'multi-scale_features'
        })
    elif backbone_name.startswith('densenet'):
        info.update({
            'architecture': 'CNN',
            'feature_type': 'dense_connections'
        })

    return info