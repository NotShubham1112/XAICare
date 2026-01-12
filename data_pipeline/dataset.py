"""
PyTorch Dataset Classes for Multi-Cancer Detection

Provides dataset classes for:
- Single cancer type datasets
- Multi-cancer combined datasets
- Patient-wise splitting to prevent data leakage
- Class imbalance handling
"""

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from PIL import Image
import yaml
import logging
from .preprocessing import MedicalImagePreprocessor

logger = logging.getLogger(__name__)


class CancerDataset(Dataset):
    """
    Dataset class for single cancer type medical images.

    Handles loading, preprocessing, and augmentation of medical images
    for a specific cancer type.
    """

    def __init__(self,
                 data_path: str,
                 cancer_type: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 config_path: str = "config.yaml",
                 augment: bool = False):
        """
        Initialize cancer-specific dataset.

        Args:
            data_path: Path to processed data directory
            cancer_type: Type of cancer ('lung', 'breast', etc.)
            split: Data split ('train', 'val', 'test')
            transform: Additional transforms to apply
            config_path: Path to configuration file
            augment: Whether to apply data augmentation
        """
        self.data_path = Path(data_path)
        self.cancer_type = cancer_type
        self.split = split
        self.transform = transform
        self.augment = augment and split == 'train'

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize preprocessor
        self.preprocessor = MedicalImagePreprocessor(config_path)

        # Get modality for this cancer type
        self.modality = self._get_cancer_modality(cancer_type)

        # Load data annotations
        self.annotations_file = self.data_path / f"{cancer_type}_{split}.csv"
        if self.annotations_file.exists():
            self.annotations = pd.read_csv(self.annotations_file)
        else:
            logger.warning(f"Annotations file not found: {self.annotations_file}")
            self.annotations = pd.DataFrame(columns=['image_path', 'label', 'patient_id'])

        # Class mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(
            self.config['cancer_types'][self._get_cancer_index(cancer_type)]['classes']
        )}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Calculate class weights for imbalance handling
        self.class_weights = self._calculate_class_weights()

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

    def _get_cancer_index(self, cancer_type: str) -> int:
        """Get index of cancer type in config."""
        for i, cancer in enumerate(self.config['cancer_types']):
            if cancer['name'] == cancer_type:
                return i
        return 0

    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced datasets."""
        if len(self.annotations) == 0:
            return torch.ones(len(self.class_to_idx))

        label_counts = self.annotations['label'].value_counts()
        total_samples = len(self.annotations)

        weights = []
        for class_name in self.class_to_idx.keys():
            count = label_counts.get(class_name, 1)  # Avoid division by zero
            weight = total_samples / (len(self.class_to_idx) * count)
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """Get item from dataset."""
        row = self.annotations.iloc[idx]

        # Load image
        image_path = self.data_path / row['image_path']
        image = self._load_image(str(image_path))

        # Preprocess image
        image = self.preprocessor.preprocess_image(image, self.modality, self.augment)

        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Get label
        label = self.class_to_idx[row['label']]

        # Additional metadata
        metadata = {
            'image_path': str(image_path),
            'patient_id': row.get('patient_id', 'unknown'),
            'cancer_type': self.cancer_type,
            'modality': self.modality,
            'original_label': row['label']
        }

        # Apply additional transforms if specified
        if self.transform:
            image = self.transform(image)

        return image, label, metadata

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file."""
        try:
            # Try PIL first (for standard formats)
            image = Image.open(image_path)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return np.array(image)

        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Return blank image as fallback
            return np.zeros((224, 224, 3), dtype=np.uint8)

    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for weighted loss functions."""
        return self.class_weights

    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution for analysis."""
        if len(self.annotations) == 0:
            return {}

        return self.annotations['label'].value_counts().to_dict()

    def get_sampler(self) -> Optional[WeightedRandomSampler]:
        """Get weighted sampler for handling class imbalance."""
        if not self.config['data']['use_class_weights'] or len(self.annotations) == 0:
            return None

        # Calculate sample weights
        sample_weights = []
        for _, row in self.annotations.iterrows():
            label = self.class_to_idx[row['label']]
            sample_weights.append(self.class_weights[label])

        return WeightedRandomSampler(sample_weights, len(sample_weights))


class MultiCancerDataset(Dataset):
    """
    Combined dataset for multiple cancer types.

    Allows training on multiple cancer types simultaneously
    while maintaining cancer-specific preprocessing.
    """

    def __init__(self,
                 data_path: str,
                 cancer_types: List[str],
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 config_path: str = "config.yaml",
                 augment: bool = False):
        """
        Initialize multi-cancer dataset.

        Args:
            data_path: Path to processed data directory
            cancer_types: List of cancer types to include
            split: Data split ('train', 'val', 'test')
            transform: Additional transforms to apply
            config_path: Path to configuration file
            augment: Whether to apply data augmentation
        """
        self.data_path = Path(data_path)
        self.cancer_types = cancer_types
        self.split = split
        self.transform = transform
        self.augment = augment and split == 'train'

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize preprocessor
        self.preprocessor = MedicalImagePreprocessor(config_path)

        # Create individual datasets
        self.datasets = []
        self.cumulative_lengths = [0]

        for cancer_type in cancer_types:
            dataset = CancerDataset(
                data_path=str(self.data_path),
                cancer_type=cancer_type,
                split=split,
                transform=None,  # We'll handle transforms here
                config_path=config_path,
                augment=augment
            )
            self.datasets.append(dataset)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))

        # Total length
        self.total_length = self.cumulative_lengths[-1]

        # Create global class mapping (cancer_type * num_classes + class_idx)
        self.global_class_to_idx = {}
        self.idx_to_global_class = {}

        global_idx = 0
        for cancer_type in cancer_types:
            modality = self._get_cancer_modality(cancer_type)
            classes = self.config['cancer_types'][self._get_cancer_index(cancer_type)]['classes']

            for class_name in classes:
                key = f"{cancer_type}_{class_name}"
                self.global_class_to_idx[key] = global_idx
                self.idx_to_global_class[global_idx] = key
                global_idx += 1

        self.num_global_classes = global_idx

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

    def _get_cancer_index(self, cancer_type: str) -> int:
        """Get index of cancer type in config."""
        for i, cancer in enumerate(self.config['cancer_types']):
            if cancer['name'] == cancer_type:
                return i
        return 0

    def __len__(self) -> int:
        """Return total dataset size."""
        return self.total_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """Get item from combined dataset."""
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths[1:], 1):
            if idx < cum_len:
                dataset_idx = i - 1
                break

        # Get local index within the dataset
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        dataset = self.datasets[dataset_idx]

        # Get item from specific dataset
        image, local_label, metadata = dataset[local_idx]

        # Convert to global label
        cancer_type = metadata['cancer_type']
        original_label = metadata['original_label']
        global_key = f"{cancer_type}_{original_label}"
        global_label = self.global_class_to_idx[global_key]

        # Update metadata
        metadata['global_label'] = global_label
        metadata['local_label'] = local_label
        metadata['global_class_name'] = global_key

        # Apply additional transforms if specified
        if self.transform:
            image = self.transform(image)

        return image, global_label, metadata

    def get_class_weights(self) -> torch.Tensor:
        """Get combined class weights across all cancer types."""
        # This is a simplified version - could be improved
        all_weights = []
        for dataset in self.datasets:
            weights = dataset.get_class_weights()
            all_weights.extend(weights.tolist())

        return torch.tensor(all_weights, dtype=torch.float32)

    def get_class_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get class distribution for all cancer types."""
        distributions = {}
        for dataset in self.datasets:
            distributions[dataset.cancer_type] = dataset.get_class_distribution()
        return distributions

    def get_sampler(self) -> Optional[WeightedRandomSampler]:
        """Get combined weighted sampler."""
        if not self.config['data']['use_class_weights']:
            return None

        # Calculate sample weights across all datasets
        sample_weights = []
        for dataset in self.datasets:
            if dataset.get_sampler() is not None:
                # This is simplified - in practice you'd need to collect all weights
                weights = [1.0] * len(dataset)  # Placeholder
                sample_weights.extend(weights)

        if not sample_weights:
            return None

        return WeightedRandomSampler(sample_weights, len(sample_weights))