"""
Data Augmentation for Medical Imaging

Provides augmentation pipelines suitable for medical images:
- Medical-image-safe transformations
- Modality-specific augmentations
- Balanced augmentation strategies
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from typing import Dict, List, Optional, Union, Callable
import numpy as np
import yaml
import logging

logger = logging.getLogger(__name__)


class MedicalAugmentation:
    """
    Medical imaging augmentation pipeline.

    Designed to be safe for medical diagnosis while improving model generalization.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize augmentation with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.aug_config = self.config['data']['augmentation']

    def get_augmentation_transform(self, modality: str = 'general') -> T.Compose:
        """
        Get augmentation transforms for specific modality.

        Args:
            modality: Imaging modality ('CT', 'MRI', 'mammogram', 'histopathology', etc.)

        Returns:
            Composed augmentation transforms
        """
        transforms = []

        # Basic augmentations (safe for most medical imaging)
        if self.aug_config['horizontal_flip']:
            transforms.append(T.RandomHorizontalFlip(p=0.5))

        # Vertical flip is usually disabled for medical images
        # as anatomical orientation matters
        if self.aug_config['vertical_flip']:
            transforms.append(T.RandomVerticalFlip(p=0.5))

        # Random rotation
        if self.aug_config['rotation_range'] > 0:
            transforms.append(T.RandomRotation(
                degrees=self.aug_config['rotation_range'],
                fill=0
            ))

        # Brightness/contrast adjustment
        if self.aug_config['brightness_contrast'] > 0:
            transforms.append(T.ColorJitter(
                brightness=self.aug_config['brightness_contrast'],
                contrast=self.aug_config['brightness_contrast']
            ))

        # Gaussian noise (subtle, medical-safe)
        if self.aug_config['gaussian_noise'] > 0:
            transforms.append(GaussianNoise(
                std=self.aug_config['gaussian_noise']
            ))

        # Modality-specific augmentations
        if modality == 'CT':
            transforms.extend(self._get_ct_augmentations())
        elif modality == 'MRI':
            transforms.extend(self._get_mri_augmentations())
        elif modality == 'mammogram':
            transforms.extend(self._get_mammogram_augmentations())
        elif modality == 'histopathology':
            transforms.extend(self._get_histopathology_augmentations())

        # Convert to tensor at the end
        transforms.append(T.ToTensor())

        return T.Compose(transforms)

    def _get_ct_augmentations(self) -> List[T.Transform]:
        """CT-specific augmentations."""
        return [
            # Subtle intensity variations (CT numbers are standardized)
            RandomIntensityShift(shift_range=(-50, 50)),
        ]

    def _get_mri_augmentations(self) -> List[T.Transform]:
        """MRI-specific augmentations."""
        return [
            # T1/T2 weighted specific augmentations could be added here
            RandomBiasField(),  # Simulate coil bias
        ]

    def _get_mammogram_augmentations(self) -> List[T.Transform]:
        """Mammogram-specific augmentations."""
        return [
            # Mammogram-specific transformations
            # Avoid extreme rotations that might confuse left/right breast
        ]

    def _get_histopathology_augmentations(self) -> List[T.Transform]:
        """Histopathology-specific augmentations."""
        return [
            # Stain normalization variations
            RandomStainVariation(),
        ]

    def get_validation_transform(self) -> T.Compose:
        """Get transforms for validation/test (no augmentation)."""
        return T.Compose([
            T.ToTensor(),
        ])


class GaussianNoise(torch.nn.Module):
    """Add Gaussian noise to images."""

    def __init__(self, std: float = 0.05):
        super().__init__()
        self.std = std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std > 0:
            noise = torch.randn_like(tensor) * self.std
            return tensor + noise
        return tensor


class RandomIntensityShift(torch.nn.Module):
    """Randomly shift image intensities."""

    def __init__(self, shift_range: Tuple[float, float] = (-0.1, 0.1)):
        super().__init__()
        self.shift_range = shift_range

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.training:
            shift = torch.FloatTensor(1).uniform_(*self.shift_range)
            return tensor + shift
        return tensor


class RandomBiasField(torch.nn.Module):
    """Simulate MRI bias field inhomogeneity."""

    def __init__(self, max_strength: float = 0.3):
        super().__init__()
        self.max_strength = max_strength

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.training and tensor.dim() == 3:  # CHW format
            # Create simple bias field
            c, h, w = tensor.shape
            bias_field = torch.ones(h, w, device=tensor.device)

            # Add some spatial variation
            strength = torch.rand(1) * self.max_strength
            bias_field = bias_field * (1 + strength * torch.rand(h, w, device=tensor.device))

            return tensor * bias_field.unsqueeze(0)
        return tensor


class RandomStainVariation(torch.nn.Module):
    """Simulate stain variation in histopathology images."""

    def __init__(self, max_variation: float = 0.2):
        super().__init__()
        self.max_variation = max_variation

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.training and tensor.dim() == 3 and tensor.shape[0] >= 3:
            # Apply random stain variation to RGB channels
            variation = torch.rand(3, device=tensor.device) * 2 * self.max_variation - self.max_variation
            stain_factors = 1 + variation

            # Apply per-channel scaling
            for c in range(3):
                tensor[c] *= stain_factors[c]

            return torch.clamp(tensor, 0, 1)
        return tensor


def get_medical_augmentation(modality: str = 'general',
                           config_path: str = "config.yaml") -> T.Compose:
    """
    Convenience function to get medical augmentation transforms.

    Args:
        modality: Imaging modality
        config_path: Path to configuration file

    Returns:
        Composed augmentation transforms
    """
    augmenter = MedicalAugmentation(config_path)
    return augmenter.get_augmentation_transform(modality)


def get_validation_transforms() -> T.Compose:
    """Get validation transforms (no augmentation)."""
    return T.Compose([
        T.ToTensor(),
    ])


class MedicalImageTransforms:
    """
    Advanced medical image transformation pipeline.

    Includes elastic deformations, advanced augmentations,
    and medical-image-safe transformations.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize advanced transforms."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_strong_augmentation(self, modality: str = 'general') -> T.Compose:
        """Get stronger augmentation for data-limited scenarios."""
        transforms = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),  # Limited for medical
            T.RandomRotation(degrees=30, fill=0),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            GaussianNoise(std=0.1),
            RandomIntensityShift(shift_range=(-0.2, 0.2)),
        ]

        # Modality-specific strong augmentations
        if modality == 'histopathology':
            transforms.extend([
                RandomStainVariation(max_variation=0.3),
                T.RandomCrop(size=(192, 192), padding=32),  # Random crop with padding
            ])

        transforms.append(T.ToTensor())
        return T.Compose(transforms)

    def get_weak_augmentation(self, modality: str = 'general') -> T.Compose:
        """Get weaker augmentation for consistency regularization."""
        transforms = [
            T.RandomHorizontalFlip(p=0.3),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            GaussianNoise(std=0.05),
        ]

        transforms.append(T.ToTensor())
        return T.Compose(transforms)


def create_augmentation_pipeline(augmentation_type: str = 'basic',
                               modality: str = 'general',
                               config_path: str = "config.yaml") -> T.Compose:
    """
    Create augmentation pipeline based on type.

    Args:
        augmentation_type: Type of augmentation ('basic', 'strong', 'weak', 'validation')
        modality: Imaging modality
        config_path: Path to configuration file

    Returns:
        Composed transforms
    """
    transforms = MedicalImageTransforms(config_path)

    if augmentation_type == 'basic':
        return get_medical_augmentation(modality, config_path)
    elif augmentation_type == 'strong':
        return transforms.get_strong_augmentation(modality)
    elif augmentation_type == 'weak':
        return transforms.get_weak_augmentation(modality)
    elif augmentation_type == 'validation':
        return get_validation_transforms()
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")