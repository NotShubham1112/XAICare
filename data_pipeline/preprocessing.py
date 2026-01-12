"""
Medical Image Preprocessing Module

Provides preprocessing pipelines for different medical imaging modalities:
- CT scans, MRIs, mammograms, histopathology, dermoscopy
- Normalization, resizing, augmentation
- Organ-specific preprocessing
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


class MedicalImagePreprocessor:
    """
    Medical image preprocessing with modality-specific pipelines.

    Supports multiple imaging modalities with appropriate preprocessing:
    - CT/MRI: Windowing, noise reduction
    - Mammograms: Contrast enhancement, pectoral muscle removal
    - Histopathology: Stain normalization
    - Dermoscopy: Hair removal, color normalization
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.target_size = tuple(self.config['model']['input_size'])
        self.normalization_mean = np.array(self.config['data']['normalization_mean'])
        self.normalization_std = np.array(self.config['data']['normalization_std'])

        # Modality-specific preprocessing parameters
        self.modality_params = {
            'CT': {
                'window_center': 40,
                'window_width': 400,
                'clip_limit': 2.0
            },
            'MRI': {
                'window_center': 0,
                'window_width': 1.0,
                'clip_limit': 2.0
            },
            'mammogram': {
                'clip_limit': 2.0,
                'grid_size': 8
            },
            'histopathology': {
                'stain_norm_method': 'macenko'
            },
            'dermoscopy': {
                'hair_removal': True,
                'color_norm': 'grayworld'
            }
        }

    def preprocess_image(self,
                        image: np.ndarray,
                        modality: str,
                        augment: bool = False) -> np.ndarray:
        """
        Main preprocessing pipeline for medical images.

        Args:
            image: Input image array
            modality: Imaging modality ('CT', 'MRI', 'mammogram', etc.)
            augment: Whether to apply data augmentation

        Returns:
            Preprocessed image array
        """
        # Convert to float32 for processing
        image = image.astype(np.float32)

        # Modality-specific preprocessing
        if modality in ['CT', 'MRI']:
            image = self._preprocess_ct_mri(image, modality)
        elif modality == 'mammogram':
            image = self._preprocess_mammogram(image)
        elif modality == 'histopathology':
            image = self._preprocess_histopathology(image)
        elif modality == 'dermoscopy':
            image = self._preprocess_dermoscopy(image)

        # Standard preprocessing steps
        image = self._resize_image(image, self.target_size)
        image = self._normalize_image(image)

        if augment:
            image = self._apply_augmentation(image)

        return image

    def _preprocess_ct_mri(self, image: np.ndarray, modality: str) -> np.ndarray:
        """Preprocess CT or MRI images with windowing and noise reduction."""
        params = self.modality_params[modality]

        # Apply windowing
        window_center = params['window_center']
        window_width = params['window_width']

        img_min = window_center - window_width / 2
        img_max = window_center + window_width / 2

        image = np.clip(image, img_min, img_max)
        image = (image - img_min) / (img_max - img_min)

        # Noise reduction using bilateral filter
        image = cv2.bilateralFilter(image.astype(np.float32), 9, 75, 75)

        return image

    def _preprocess_mammogram(self, image: np.ndarray) -> np.ndarray:
        """Preprocess mammogram images with contrast enhancement."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(
            clipLimit=self.modality_params['mammogram']['clip_limit'],
            tileGridSize=(self.modality_params['mammogram']['grid_size'],
                          self.modality_params['mammogram']['grid_size'])
        )
        image = clahe.apply(image.astype(np.uint8))

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        return image

    def _preprocess_histopathology(self, image: np.ndarray) -> np.ndarray:
        """Preprocess histopathology images with stain normalization."""
        # For now, apply basic normalization
        # TODO: Implement advanced stain normalization (Macenko, Reinhard)

        # Basic stain normalization using color deconvolution
        if len(image.shape) == 3:
            # Convert RGB to optical density
            image = -np.log((image.astype(np.float32) + 1) / 256)

            # Normalize each channel
            for c in range(3):
                channel = image[:, :, c]
                image[:, :, c] = (channel - np.mean(channel)) / (np.std(channel) + 1e-8)

        return image

    def _preprocess_dermoscopy(self, image: np.ndarray) -> np.ndarray:
        """Preprocess dermoscopic images with hair removal and color normalization."""
        # Hair removal using morphological operations
        if self.modality_params['dermoscopy']['hair_removal']:
            image = self._remove_hair(image)

        # Color normalization
        if self.modality_params['dermoscopy']['color_norm'] == 'grayworld':
            image = self._grayworld_color_norm(image)

        return image

    def _remove_hair(self, image: np.ndarray) -> np.ndarray:
        """Remove hair artifacts from dermoscopic images."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Black hat filtering to detect dark hairs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Threshold and inpaint
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        image = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)

        return image

    def _grayworld_color_norm(self, image: np.ndarray) -> np.ndarray:
        """Apply gray world color normalization."""
        # Calculate mean for each channel
        mean_r = np.mean(image[:, :, 0])
        mean_g = np.mean(image[:, :, 1])
        mean_b = np.mean(image[:, :, 2])

        # Calculate overall mean
        mean_overall = (mean_r + mean_g + mean_b) / 3

        # Normalize each channel
        image[:, :, 0] = image[:, :, 0] * (mean_overall / (mean_r + 1e-8))
        image[:, :, 1] = image[:, :, 1] * (mean_overall / (mean_g + 1e-8))
        image[:, :, 2] = image[:, :, 2] * (mean_overall / (mean_b + 1e-8))

        return np.clip(image, 0, 255).astype(np.uint8)

    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        h, w = image.shape[:2]
        target_h, target_w = target_size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image
        if len(image.shape) == 3:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create canvas and center image
        canvas = np.zeros((target_h, target_w) + image.shape[2:], dtype=image.dtype)

        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image

        return canvas

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Apply ImageNet-style normalization."""
        # Ensure 3-channel image
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)

        # Normalize
        image = (image - self.normalization_mean) / self.normalization_std

        return image

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation for training."""
        aug_config = self.config['data']['augmentation']

        # Random rotation
        if aug_config['rotation_range'] > 0:
            angle = np.random.uniform(-aug_config['rotation_range'],
                                    aug_config['rotation_range'])
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            image = cv2.warpAffine(image, M, (w, h))

        # Random horizontal flip
        if aug_config['horizontal_flip'] and np.random.random() < 0.5:
            image = cv2.flip(image, 1)

        # Random vertical flip (usually disabled for medical images)
        if aug_config['vertical_flip'] and np.random.random() < 0.5:
            image = cv2.flip(image, 0)

        # Random brightness/contrast
        if aug_config['brightness_contrast'] > 0:
            alpha = 1 + np.random.uniform(-aug_config['brightness_contrast'],
                                         aug_config['brightness_contrast'])
            beta = np.random.uniform(-aug_config['brightness_contrast'] * 50,
                                   aug_config['brightness_contrast'] * 50)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Add Gaussian noise
        if aug_config['gaussian_noise'] > 0 and np.random.random() < 0.5:
            noise = np.random.normal(0, aug_config['gaussian_noise'], image.shape)
            image = np.clip(image + noise, 0, 1)

        return image

    def get_preprocessing_stats(self) -> Dict:
        """Get preprocessing statistics for monitoring."""
        return {
            'target_size': self.target_size,
            'normalization_mean': self.normalization_mean.tolist(),
            'normalization_std': self.normalization_std.tolist(),
            'supported_modalities': list(self.modality_params.keys())
        }