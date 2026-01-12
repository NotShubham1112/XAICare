"""
Cancer-Specific Classification Heads

Provides specialized classification heads for different cancer types:
- Binary classification (benign/malignant)
- Multi-class classification where applicable
- Specialized architectures for different modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import yaml
import logging

logger = logging.getLogger(__name__)


class CancerClassificationHead(nn.Module):
    """
    Classification head for a single cancer type.

    Supports binary and multi-class classification with optional
    auxiliary outputs and uncertainty estimation.
    """

    def __init__(self,
                 input_dim: int,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3,
                 use_auxiliary: bool = False,
                 config_path: str = "config.yaml"):
        """
        Initialize cancer classification head.

        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            use_auxiliary: Whether to include auxiliary outputs
            config_path: Path to configuration file
        """
        super().__init__()

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_auxiliary = use_auxiliary

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, num_classes)
        )

        # Auxiliary head for multi-task learning (optional)
        if use_auxiliary:
            self.auxiliary = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),  # Binary auxiliary task
                nn.Sigmoid()
            )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through classification head.

        Args:
            x: Input feature tensor

        Returns:
            Dictionary containing logits, probabilities, and auxiliary outputs
        """
        # Main classification
        logits = self.classifier(x)
        probabilities = F.softmax(logits, dim=-1)

        outputs = {
            'logits': logits,
            'probabilities': probabilities,
            'prediction': torch.argmax(logits, dim=-1)
        }

        # Add auxiliary outputs if enabled
        if self.use_auxiliary:
            aux_output = self.auxiliary(x).squeeze(-1)
            outputs['auxiliary'] = aux_output

        # Add uncertainty estimation (entropy)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)
        outputs['uncertainty'] = entropy

        return outputs

    def get_classification_weights(self) -> torch.Tensor:
        """Get the final classification layer weights for Grad-CAM."""
        return self.classifier[-1].weight


class MultiCancerHeads(nn.Module):
    """
    Multi-head architecture for multiple cancer types.

    Each cancer type gets its own specialized classification head
    while sharing the backbone feature extractor.
    """

    def __init__(self,
                 input_dim: int,
                 cancer_configs: List[Dict],
                 dropout_rate: float = 0.3,
                 config_path: str = "config.yaml"):
        """
        Initialize multi-cancer classification heads.

        Args:
            input_dim: Dimension of input features from backbone
            cancer_configs: List of cancer type configurations
            dropout_rate: Dropout rate for regularization
            config_path: Path to configuration file
        """
        super().__init__()

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.input_dim = input_dim
        self.cancer_configs = cancer_configs
        self.dropout_rate = dropout_rate

        # Create classification heads for each cancer type
        self.heads = nn.ModuleDict()

        for cancer_config in cancer_configs:
            cancer_name = cancer_config['name']
            num_classes = len(cancer_config['classes'])

            head = CancerClassificationHead(
                input_dim=input_dim,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                use_auxiliary=False,  # Can be enabled per cancer type if needed
                config_path=config_path
            )

            self.heads[cancer_name] = head

        # Store cancer type information
        self.cancer_names = [config['name'] for config in cancer_configs]
        self.cancer_classes = {config['name']: config['classes'] for config in cancer_configs}

        logger.info(f"Created multi-cancer heads for: {self.cancer_names}")

    def forward(self, x: torch.Tensor, cancer_type: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through appropriate classification head(s).

        Args:
            x: Input feature tensor
            cancer_type: Specific cancer type to classify (None for all)

        Returns:
            Dictionary containing predictions for requested cancer type(s)
        """
        if cancer_type is not None:
            # Single cancer type classification
            if cancer_type not in self.heads:
                raise ValueError(f"Unknown cancer type: {cancer_type}")

            head = self.heads[cancer_type]
            outputs = head(x)

            # Add cancer type information
            outputs['cancer_type'] = cancer_type
            outputs['classes'] = self.cancer_classes[cancer_type]

            return outputs

        else:
            # Multi-cancer classification (all heads)
            all_outputs = {}

            for cancer_name, head in self.heads.items():
                cancer_outputs = head(x)
                # Prefix keys to avoid conflicts
                prefixed_outputs = {f"{cancer_name}_{k}": v for k, v in cancer_outputs.items()}
                all_outputs.update(prefixed_outputs)

            all_outputs['cancer_types'] = self.cancer_names
            return all_outputs

    def get_head(self, cancer_type: str) -> CancerClassificationHead:
        """Get classification head for specific cancer type."""
        if cancer_type not in self.heads:
            raise ValueError(f"Unknown cancer type: {cancer_type}")
        return self.heads[cancer_type]

    def get_classification_weights(self, cancer_type: str) -> torch.Tensor:
        """Get classification weights for Grad-CAM for specific cancer type."""
        head = self.get_head(cancer_type)
        return head.get_classification_weights()

    def freeze_head(self, cancer_type: str):
        """Freeze parameters of specific cancer head."""
        head = self.get_head(cancer_type)
        for param in head.parameters():
            param.requires_grad = False
        logger.info(f"Froze classification head for {cancer_type}")

    def unfreeze_head(self, cancer_type: str):
        """Unfreeze parameters of specific cancer head."""
        head = self.get_head(cancer_type)
        for param in head.parameters():
            param.requires_grad = True
        logger.info(f"Unfroze classification head for {cancer_type}")

    def get_trainable_parameters(self, cancer_type: Optional[str] = None) -> List[torch.Tensor]:
        """Get trainable parameters for specific cancer type or all."""
        if cancer_type is not None:
            head = self.get_head(cancer_type)
            return list(head.parameters())
        else:
            return list(self.parameters())


class SegmentationHead(nn.Module):
    """
    Optional segmentation head for tumor localization.

    Uses U-Net style architecture for pixel-level segmentation
    of tumors/abnormalities in medical images.
    """

    def __init__(self,
                 input_dim: int,
                 num_classes: int = 2,
                 config_path: str = "config.yaml"):
        """
        Initialize segmentation head.

        Args:
            input_dim: Dimension of input features
            num_classes: Number of segmentation classes (usually 2: background/tumor)
            config_path: Path to configuration file
        """
        super().__init__()

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Simple segmentation head (can be extended to full U-Net)
        self.segmentation = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through segmentation head.

        Args:
            x: Input feature tensor (should be spatial, not flattened)

        Returns:
            Segmentation logits
        """
        return self.segmentation(x)


def create_cancer_head(cancer_config: Dict,
                      input_dim: int,
                      config_path: str = "config.yaml") -> CancerClassificationHead:
    """
    Factory function to create appropriate classification head for cancer type.

    Args:
        cancer_config: Configuration dictionary for cancer type
        input_dim: Input feature dimension
        config_path: Path to configuration file

    Returns:
        Configured classification head
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    num_classes = len(cancer_config['classes'])
    dropout_rate = config['model']['dropout_rate']

    # Create head with cancer-specific parameters if needed
    head = CancerClassificationHead(
        input_dim=input_dim,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        use_auxiliary=False,  # Can be extended based on cancer type
        config_path=config_path
    )

    return head