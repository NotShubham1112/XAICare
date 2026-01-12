"""
Grad-CAM Implementation for Visual Explanations

Provides Grad-CAM and Grad-CAM++ for visualizing discriminative regions
in medical images that contribute to cancer detection decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import cv2
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Hook-based feature extractor for Grad-CAM.

    Extracts feature maps and gradients from target layers.
    """

    def __init__(self, model: nn.Module, target_layers: List[str]):
        """
        Initialize feature extractor.

        Args:
            model: PyTorch model
            target_layers: List of layer names to extract features from
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.features = []

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layers."""
        def forward_hook(module, input, output):
            self.features.append(output)

        def backward_hook(module, grad_in, grad_out):
            self.gradients.append(grad_out[0])

        # Find target layers in model
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_layers):
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def clear_hooks(self):
        """Clear stored features and gradients."""
        self.gradients = []
        self.features = []


class GradCAM:
    """
    Grad-CAM implementation for generating visual explanations.

    Produces heatmaps showing regions of the image that contribute
    most to the model's prediction.
    """

    def __init__(self, model: nn.Module, target_layer: str = "layer4"):
        """
        Initialize Grad-CAM.

        Args:
            model: PyTorch model
            target_layer: Target layer for feature extraction
        """
        self.model = model
        self.target_layer = target_layer
        self.device = next(model.parameters()).device

        # Initialize feature extractor
        self.extractor = FeatureExtractor(model, [target_layer])

    def generate_heatmap(self,
                        input_tensor: torch.Tensor,
                        target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input image.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index (None for predicted class)

        Returns:
            Heatmap as numpy array [H, W]
        """
        self.model.eval()

        # Clear previous hooks
        self.extractor.clear_hooks()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output['prediction'].item()

        # Get logits for target class
        logits = output['logits']
        target_logit = logits[0, target_class]

        # Backward pass
        self.model.zero_grad()
        target_logit.backward(retain_graph=True)

        # Get gradients and features
        gradients = self.extractor.gradients[-1]  # [1, C, H, W]
        features = self.extractor.features[-1]    # [1, C, H, W]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of feature maps
        heatmap = torch.sum(weights * features, dim=1, keepdim=True)  # [1, 1, H, W]

        # ReLU to focus on positive contributions
        heatmap = F.relu(heatmap)

        # Normalize to [0, 1]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # Convert to numpy
        heatmap = heatmap.squeeze().cpu().detach().numpy()

        # Resize to input image size
        input_size = (input_tensor.shape[2], input_tensor.shape[3])
        heatmap = cv2.resize(heatmap, input_size)

        return heatmap

    def overlay_heatmap(self,
                       image: np.ndarray,
                       heatmap: np.ndarray,
                       alpha: float = 0.5,
                       colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Args:
            image: Original image [H, W, 3] or [H, W]
            heatmap: Grad-CAM heatmap [H, W]
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap

        Returns:
            Overlayed image [H, W, 3]
        """
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Normalize image to [0, 255] if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), colormap
        )

        # Overlay heatmap on image
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation with improved gradient weighting.

    Provides more precise localization than standard Grad-CAM.
    """

    def generate_heatmap(self,
                        input_tensor: torch.Tensor,
                        target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index (None for predicted class)

        Returns:
            Heatmap as numpy array [H, W]
        """
        self.model.eval()

        # Clear previous hooks
        self.extractor.clear_hooks()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output['prediction'].item()

        # Get logits for target class
        logits = output['logits']
        target_logit = logits[0, target_class]

        # Backward pass
        self.model.zero_grad()
        target_logit.backward(retain_graph=True)

        # Get gradients and features
        gradients = self.extractor.gradients[-1]  # [1, C, H, W]
        features = self.extractor.features[-1]    # [1, C, H, W]

        # Grad-CAM++ weighting
        gradients_2 = gradients ** 2
        gradients_3 = gradients ** 3

        # Global average pool of different gradient powers
        alpha_num = gradients_2
        alpha_denom = 2 * gradients_2 + torch.sum(
            features * gradients_3, dim=(2, 3), keepdim=True
        )
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))

        alphas = alpha_num / alpha_denom  # [1, C, H, W]

        # Global average pool of alphas
        weights = torch.mean(alphas, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination
        heatmap = torch.sum(weights * features, dim=1, keepdim=True)  # [1, 1, H, W]

        # ReLU
        heatmap = F.relu(heatmap)

        # Normalize
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # Convert to numpy
        heatmap = heatmap.squeeze().cpu().detach().numpy()

        # Resize to input size
        input_size = (input_tensor.shape[2], input_tensor.shape[3])
        heatmap = cv2.resize(heatmap, input_size)

        return heatmap


class GuidedBackpropagation:
    """
    Guided Backpropagation for pixel-level saliency.

    Combines gradients with ReLU activations for sharper visualizations.
    """

    def __init__(self, model: nn.Module):
        """Initialize Guided Backpropagation."""
        self.model = model
        self.gradients = []
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks for guided backpropagation."""
        def backward_hook(module, grad_in, grad_out):
            # Only keep positive gradients (guided)
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        # Register hooks on all ReLU layers
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(backward_hook)

    def generate_saliency(self,
                         input_tensor: torch.Tensor,
                         target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate guided backpropagation saliency map.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index

        Returns:
            Saliency map [H, W, 3]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output['prediction'].item()

        # Get target logit
        logits = output['logits']
        target_logit = logits[0, target_class]

        # Backward pass
        self.model.zero_grad()
        target_logit.backward()

        # Get gradients from input
        saliency = input_tensor.grad.data.abs()
        saliency = saliency.squeeze().cpu().numpy()

        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        # Convert to RGB if grayscale
        if saliency.shape[0] == 1:
            saliency = np.repeat(saliency, 3, axis=0)

        # Transpose to [H, W, C]
        saliency = saliency.transpose(1, 2, 0)

        return saliency


class XAIInterpreter:
    """
    Unified XAI interpreter combining multiple explanation methods.
    """

    def __init__(self, model: nn.Module, target_layer: str = "layer4"):
        """
        Initialize XAI interpreter.

        Args:
            model: PyTorch model
            target_layer: Target layer for Grad-CAM
        """
        self.model = model
        self.grad_cam = GradCAM(model, target_layer)
        self.grad_cam_pp = GradCAMPlusPlus(model, target_layer)
        self.guided_bp = GuidedBackpropagation(model)

    def explain_prediction(self,
                          input_tensor: torch.Tensor,
                          original_image: np.ndarray,
                          cancer_type: str,
                          method: str = "gradcam") -> Dict[str, np.ndarray]:
        """
        Generate comprehensive explanation for a prediction.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            original_image: Original image for overlay [H, W, 3]
            cancer_type: Cancer type for context
            method: Explanation method ('gradcam', 'gradcam++', 'guided')

        Returns:
            Dictionary containing heatmaps and overlays
        """
        explanations = {}

        # Generate primary heatmap
        if method == "gradcam":
            heatmap = self.grad_cam.generate_heatmap(input_tensor)
        elif method == "gradcam++":
            heatmap = self.grad_cam_pp.generate_heatmap(input_tensor)
        else:
            raise ValueError(f"Unknown method: {method}")

        explanations['heatmap'] = heatmap

        # Generate overlay
        overlay = self.grad_cam.overlay_heatmap(original_image, heatmap)
        explanations['overlay'] = overlay

        # Generate guided backpropagation saliency
        saliency = self.guided_bp.generate_saliency(input_tensor)
        explanations['saliency'] = saliency

        return explanations

    def get_attention_regions(self, heatmap: np.ndarray, threshold: float = 0.5) -> Dict:
        """
        Extract attention regions from heatmap.

        Args:
            heatmap: Grad-CAM heatmap [H, W]
            threshold: Threshold for region extraction

        Returns:
            Dictionary with region information
        """
        # Threshold heatmap
        binary_map = (heatmap > threshold).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            regions.append({
                'bbox': (x, y, w, h),
                'area': area,
                'center': (x + w//2, y + h//2)
            })

        # Sort by area (largest first)
        regions.sort(key=lambda x: x['area'], reverse=True)

        return {
            'num_regions': len(regions),
            'regions': regions,
            'total_attention_area': np.sum(binary_map),
            'attention_percentage': np.sum(binary_map) / (heatmap.shape[0] * heatmap.shape[1])
        }


def save_explanation_visualization(explanations: Dict[str, np.ndarray],
                                 save_path: str,
                                 filename_prefix: str = "explanation"):
    """
    Save explanation visualizations to disk.

    Args:
        explanations: Dictionary of explanation images
        save_path: Directory to save images
        filename_prefix: Prefix for saved files
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for name, image in explanations.items():
        filepath = save_path / f"{filename_prefix}_{name}.png"
        cv2.imwrite(str(filepath), cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image)
        logger.info(f"Saved {name} to {filepath}")


def create_explanation_report(explanations: Dict,
                            prediction: Dict,
                            cancer_type: str) -> Dict:
    """
    Create comprehensive explanation report.

    Args:
        explanations: XAI explanations dictionary
        prediction: Model prediction results
        cancer_type: Cancer type

    Returns:
        Explanation report dictionary
    """
    attention_info = explanations.get('attention_regions', {})

    report = {
        'cancer_type': cancer_type,
        'prediction': prediction,
        'attention_analysis': attention_info,
        'visual_explanations': {
            'has_heatmap': 'heatmap' in explanations,
            'has_overlay': 'overlay' in explanations,
            'has_saliency': 'saliency' in explanations
        },
        'clinical_interpretation': {
            'attention_regions_count': attention_info.get('num_regions', 0),
            'attention_coverage': attention_info.get('attention_percentage', 0),
            'confidence_score': prediction.get('confidence', 0)
        }
    }

    return report