"""
Saliency Map Generation for Model Interpretability

Provides various saliency methods for understanding model decisions:
- Vanilla gradients
- SmoothGrad
- Integrated Gradients
- SHAP values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import cv2
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


class SaliencyMapGenerator:
    """
    Generate saliency maps using various attribution methods.

    Provides multiple techniques for understanding pixel-level
    contributions to model predictions.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize saliency generator.

        Args:
            model: PyTorch model
        """
        self.model = model
        self.device = next(model.parameters()).device

    def generate_vanilla_gradients(self,
                                  input_tensor: torch.Tensor,
                                  target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate vanilla gradient saliency map.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index

        Returns:
            Saliency map [H, W, C]
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

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

        # Get gradients
        gradients = input_tensor.grad.data.abs()

        # Convert to numpy
        saliency = gradients.squeeze().cpu().numpy()

        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        # Convert to RGB if grayscale
        if saliency.shape[0] == 1:
            saliency = np.repeat(saliency, 3, axis=0)

        # Transpose to [H, W, C]
        saliency = saliency.transpose(1, 2, 0)

        return saliency

    def generate_smoothgrad(self,
                           input_tensor: torch.Tensor,
                           target_class: Optional[int] = None,
                           num_samples: int = 50,
                           noise_std: float = 0.1) -> np.ndarray:
        """
        Generate SmoothGrad saliency map.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index
            num_samples: Number of noisy samples
            noise_std: Standard deviation of noise

        Returns:
            SmoothGrad saliency map [H, W, C]
        """
        saliency_maps = []

        for _ in range(num_samples):
            # Add noise to input
            noise = torch.randn_like(input_tensor) * noise_std
            noisy_input = input_tensor + noise

            # Generate gradient saliency
            saliency = self.generate_vanilla_gradients(noisy_input, target_class)
            saliency_maps.append(saliency)

        # Average saliency maps
        smooth_saliency = np.mean(saliency_maps, axis=0)

        return smooth_saliency

    def generate_integrated_gradients(self,
                                     input_tensor: torch.Tensor,
                                     target_class: Optional[int] = None,
                                     baseline: Optional[torch.Tensor] = None,
                                     num_steps: int = 50) -> np.ndarray:
        """
        Generate Integrated Gradients saliency map.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index
            baseline: Baseline input (zeros if None)
            num_steps: Number of integration steps

        Returns:
            Integrated Gradients saliency map [H, W, C]
        """
        self.model.eval()

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Create interpolated inputs
        alphas = torch.linspace(0, 1, num_steps).to(self.device)
        interpolated_inputs = []

        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            interpolated_inputs.append(interpolated)

        # Compute gradients at each step
        gradients = []
        for interpolated in interpolated_inputs:
            output = self.model(interpolated)

            if target_class is None:
                target_class = output['prediction'].item()

            logits = output['logits']
            target_logit = logits[0, target_class]

            self.model.zero_grad()
            target_logit.backward()

            gradients.append(interpolated.grad.data)

        # Compute integrated gradients
        gradients = torch.stack(gradients)
        avg_gradients = torch.mean(gradients, dim=0)

        integrated_gradients = (input_tensor - baseline) * avg_gradients

        # Convert to numpy
        saliency = integrated_gradients.abs().squeeze().cpu().numpy()

        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        # Convert to RGB if needed
        if saliency.shape[0] == 1:
            saliency = np.repeat(saliency, 3, axis=0)

        # Transpose to [H, W, C]
        saliency = saliency.transpose(1, 2, 0)

        return saliency

    def generate_shap_values(self,
                            input_tensor: torch.Tensor,
                            target_class: Optional[int] = None,
                            num_samples: int = 100) -> np.ndarray:
        """
        Generate SHAP values using Kernel SHAP approximation.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index
            num_samples: Number of SHAP samples

        Returns:
            SHAP attribution map [H, W, C]
        """
        # This is a simplified implementation
        # For full SHAP, consider using the shap library

        self.model.eval()
        input_np = input_tensor.squeeze().cpu().numpy()

        # Get baseline (black image)
        baseline = np.zeros_like(input_np)

        shap_values = np.zeros_like(input_np)

        for _ in range(num_samples):
            # Create random mask
            mask = np.random.binomial(1, 0.5, size=input_np.shape)

            # Apply mask
            masked_input = baseline + mask * (input_np - baseline)
            masked_tensor = torch.from_numpy(masked_input).unsqueeze(0).to(self.device).float()

            # Get model output
            with torch.no_grad():
                output = self.model(masked_tensor)

            if target_class is None:
                target_class = output['prediction'].item()

            # Get prediction for target class
            prob = output['probabilities'][0, target_class].item()

            # Update SHAP values
            shap_values += prob * (mask - 0.5) * 2  # Center around 0

        # Average
        shap_values /= num_samples

        # Take absolute value for saliency
        saliency = np.abs(shap_values)

        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        # Convert to RGB if needed
        if saliency.shape[0] == 1:
            saliency = np.repeat(saliency, 3, axis=0)

        # Transpose to [H, W, C]
        saliency = saliency.transpose(1, 2, 0)

        return saliency

    def generate_occlusion_sensitivity(self,
                                     input_tensor: torch.Tensor,
                                     target_class: Optional[int] = None,
                                     patch_size: int = 16,
                                     stride: int = 8) -> np.ndarray:
        """
        Generate occlusion sensitivity map.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index
            patch_size: Size of occlusion patch
            stride: Stride for sliding window

        Returns:
            Occlusion sensitivity map [H, W]
        """
        self.model.eval()

        _, _, H, W = input_tensor.shape
        sensitivity_map = np.zeros((H, W))

        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(input_tensor)

        if target_class is None:
            target_class = baseline_output['prediction'].item()

        baseline_prob = baseline_output['probabilities'][0, target_class].item()

        # Slide occlusion window
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                # Create occluded input
                occluded = input_tensor.clone()
                occluded[:, :, y:y+patch_size, x:x+patch_size] = 0

                # Get prediction
                with torch.no_grad():
                    output = self.model(occluded)
                    prob = output['probabilities'][0, target_class].item()

                # Compute sensitivity (drop in probability)
                sensitivity = baseline_prob - prob
                sensitivity_map[y:y+patch_size, x:x+patch_size] = sensitivity

        # Normalize to [0, 1]
        sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min() + 1e-8)

        return sensitivity_map


class AdvancedSaliencyGenerator:
    """
    Advanced saliency methods with noise injection and regularization.
    """

    def __init__(self, model: nn.Module):
        """Initialize advanced saliency generator."""
        self.model = model
        self.basic_generator = SaliencyMapGenerator(model)

    def generate_robust_saliency(self,
                                input_tensor: torch.Tensor,
                                target_class: Optional[int] = None,
                                num_perturbations: int = 10) -> np.ndarray:
        """
        Generate robust saliency map with input perturbations.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index
            num_perturbations: Number of input perturbations

        Returns:
            Robust saliency map [H, W, C]
        """
        saliency_maps = []

        for _ in range(num_perturbations):
            # Add small random perturbation
            perturbation = torch.randn_like(input_tensor) * 0.01
            perturbed_input = input_tensor + perturbation

            # Generate saliency
            saliency = self.basic_generator.generate_vanilla_gradients(
                perturbed_input, target_class
            )
            saliency_maps.append(saliency)

        # Average saliency maps
        robust_saliency = np.mean(saliency_maps, axis=0)

        return robust_saliency

    def generate_guided_saliency(self,
                                input_tensor: torch.Tensor,
                                target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate guided saliency combining gradients with feature importance.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index

        Returns:
            Guided saliency map [H, W, C]
        """
        # Get vanilla gradients
        gradients = self.basic_generator.generate_vanilla_gradients(
            input_tensor, target_class
        )

        # Get occlusion sensitivity
        occlusion = self.basic_generator.generate_occlusion_sensitivity(
            input_tensor, target_class
        )

        # Combine methods
        # Weight gradients by occlusion sensitivity
        occlusion_rgb = cv2.cvtColor(occlusion, cv2.COLOR_GRAY2RGB)
        guided_saliency = gradients * occlusion_rgb

        # Normalize
        guided_saliency = (guided_saliency - guided_saliency.min()) / (guided_saliency.max() - guided_saliency.min() + 1e-8)

        return guided_saliency


def compare_saliency_methods(model: nn.Module,
                           input_tensor: torch.Tensor,
                           target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Compare different saliency methods on the same input.

    Args:
        model: PyTorch model
        input_tensor: Input image tensor
        target_class: Target class index

    Returns:
        Dictionary of saliency maps from different methods
    """
    generator = SaliencyMapGenerator(model)

    methods = {
        'vanilla_gradients': generator.generate_vanilla_gradients,
        'smoothgrad': lambda x, c: generator.generate_smoothgrad(x, c, num_samples=25),
        'integrated_gradients': generator.generate_integrated_gradients,
        'occlusion': lambda x, c: cv2.cvtColor(
            generator.generate_occlusion_sensitivity(x, c), cv2.COLOR_GRAY2RGB
        )
    }

    results = {}
    for method_name, method_func in methods.items():
        try:
            saliency = method_func(input_tensor, target_class)
            results[method_name] = saliency
            logger.info(f"Generated {method_name} saliency map")
        except Exception as e:
            logger.warning(f"Failed to generate {method_name}: {e}")

    return results


def save_saliency_comparison(saliency_maps: Dict[str, np.ndarray],
                           save_path: str,
                           filename_prefix: str = "saliency_comparison"):
    """
    Save saliency maps from different methods for comparison.

    Args:
        saliency_maps: Dictionary of saliency maps
        save_path: Directory to save images
        filename_prefix: Prefix for saved files
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for method_name, saliency in saliency_maps.items():
        # Convert to BGR for OpenCV
        if saliency.shape[-1] == 3:
            save_img = cv2.cvtColor((saliency * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            save_img = (saliency * 255).astype(np.uint8)

        filepath = save_path / f"{filename_prefix}_{method_name}.png"
        cv2.imwrite(str(filepath), save_img)
        logger.info(f"Saved {method_name} saliency to {filepath}")