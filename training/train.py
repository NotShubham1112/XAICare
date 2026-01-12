"""
Main Training Script for Multi-Cancer AI Detection

Implements transfer learning pipeline with:
- Progressive backbone unfreezing
- Weighted loss functions for imbalanced data
- Medical imaging-specific training
- Clinical metric optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from pathlib import Path
import yaml
import argparse
import logging
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Import project modules
from models.multi_cancer_model import create_multi_cancer_model
from data_pipeline.dataset import MultiCancerDataset
from data_pipeline.preprocessing import MedicalImagePreprocessor
from training.augmentation import create_augmentation_pipeline
from training.callbacks import create_training_callbacks, EarlyStopping
from evaluation.metrics import calculate_clinical_metrics

logger = logging.getLogger(__name__)


class TransferLearningTrainer:
    """
    Trainer for transfer learning on multi-cancer detection.

    Implements progressive unfreezing, weighted loss, and clinical optimization.
    """

    def __init__(self,
                 model: nn.Module,
                 cancer_types: List[str],
                 config_path: str = "config.yaml",
                 device: str = None):
        """
        Initialize trainer.

        Args:
            model: Multi-cancer model
            cancer_types: List of cancer types to train
            config_path: Path to configuration file
            device: Device to train on ('cuda', 'cpu', etc.)
        """
        self.model = model
        self.cancer_types = cancer_types

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Training configuration
        train_config = self.config['training']
        self.batch_size = train_config['batch_size']
        self.num_epochs = train_config['num_epochs']
        self.learning_rate = train_config['learning_rate']
        self.weight_decay = train_config['weight_decay']

        # Progressive unfreezing
        self.progressive_unfreezing = train_config['progressive_unfreezing']
        self.unfreeze_layers = train_config.get('unfreeze_layers', [4, 5])

        # Loss configuration
        self.use_weighted_loss = train_config['use_weighted_loss']
        self.focal_loss_alpha = train_config.get('focal_loss_alpha', 0.25)
        self.focal_loss_gamma = train_config.get('focal_loss_gamma', 2.0)

        # Initialize components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.callbacks = None

        self._setup_training_components()

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Training {len(cancer_types)} cancer types: {cancer_types}")

    def _setup_training_components(self):
        """Setup optimizer, scheduler, loss function, and callbacks."""
        # Get parameter groups for differential learning rates
        param_groups = self.model.get_parameter_groups(
            lr_backbone=self.learning_rate * 0.1,  # Lower LR for backbone
            lr_heads=self.learning_rate
        )

        # Optimizer
        if self.config['training']['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(param_groups, weight_decay=self.weight_decay)
        elif self.config['training']['optimizer'].lower() == 'adamw':
            self.optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)
        elif self.config['training']['optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=self.weight_decay)

        # Learning rate scheduler
        scheduler_type = self.config['training']['scheduler']
        if scheduler_type == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs
            )
        elif scheduler_type == 'step':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif scheduler_type == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5, factor=0.5
            )

        # Loss function
        self.criterion = self._create_loss_function()

        # Callbacks
        self.callbacks = create_training_callbacks(self.config_path)

    def _create_loss_function(self) -> nn.Module:
        """Create appropriate loss function based on configuration."""
        if self.use_weighted_loss:
            # Get class weights from training data
            # This would be calculated from the dataset
            # For now, use focal loss as alternative
            return FocalLoss(alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma)
        else:
            return nn.CrossEntropyLoss()

    def _progressive_unfreeze(self, epoch: int):
        """Implement progressive unfreezing of backbone layers."""
        if not self.progressive_unfreezing:
            return

        # Unfreeze layers based on epoch
        layers_to_unfreeze = []
        if epoch >= self.num_epochs // 3:
            layers_to_unfreeze.extend([4])  # Unfreeze layer4
        if epoch >= (2 * self.num_epochs) // 3:
            layers_to_unfreeze.extend([5])  # Unfreeze layer5 (closest to classifier)

        if layers_to_unfreeze:
            self.model.unfreeze_backbone_layers(layers_to_unfreeze)
            logger.info(f"Epoch {epoch}: Unfroze backbone layers {layers_to_unfreeze}")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # Progressive unfreezing
        self._progressive_unfreeze(epoch)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (images, labels, metadata) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Handle multi-cancer labels (global indexing)
            outputs = self.model(images)
            loss = self.criterion(outputs['logits'], labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            epoch_loss += loss.item()
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{epoch_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })

        epoch_loss /= len(train_loader)
        epoch_acc = 100. * correct / total

        return {
            'train_loss': epoch_loss,
            'train_acc': epoch_acc
        }

    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels, metadata in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs['logits'], labels)

                epoch_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Store for clinical metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(outputs['probabilities'].cpu().numpy())

        epoch_loss /= len(val_loader)
        epoch_acc = 100. * correct / total

        # Calculate clinical metrics
        clinical_metrics = calculate_clinical_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities)
        )

        val_metrics = {
            'val_loss': epoch_loss,
            'val_acc': epoch_acc,
            'val_sensitivity': clinical_metrics['sensitivity'],
            'val_specificity': clinical_metrics['specificity'],
            'val_auc': clinical_metrics['auc'],
            'val_f1': clinical_metrics['f1_score']
        }

        return val_metrics

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              save_path: str = None) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Path to save trained model

        Returns:
            Training history
        """
        logger.info("Starting training...")
        logger.info(f"Training for {self.num_epochs} epochs on {len(self.cancer_types)} cancer types")

        # Initialize callbacks
        self.callbacks.on_train_begin()

        # Training history
        history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_sensitivity': [],
            'val_specificity': [],
            'val_auc': [],
            'val_f1': []
        }

        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.config['training']['early_stopping_metric'],
            mode='max' if 'auc' in self.config['training']['early_stopping_metric'] else 'min',
            patience=self.config['training']['early_stopping_patience']
        )

        best_model_state = None
        best_metric_value = float('-inf')

        for epoch in range(self.num_epochs):
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validation phase
            val_metrics = self.validate_epoch(val_loader, epoch)

            # Combine metrics
            epoch_logs = {**train_metrics, **val_metrics}

            # Update history
            for key in history.keys():
                if key in epoch_logs:
                    history[key].append(epoch_logs[key])

            # Callbacks
            self.callbacks.on_epoch_end(epoch, epoch_logs)

            # Learning rate scheduling
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(epoch_logs.get('val_auc', epoch_logs.get('val_loss')))
            else:
                self.scheduler.step()

            # Early stopping check
            monitor_metric = epoch_logs.get(early_stopping.monitor)
            if monitor_metric is not None:
                early_stopping.on_epoch_end(epoch, epoch_logs)

                if monitor_metric > best_metric_value:
                    best_metric_value = monitor_metric
                    best_model_state = self.model.state_dict().copy()

            if early_stopping.should_stop():
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore best model if early stopping was used
        if best_model_state is not None and early_stopping.restore_best_weights:
            self.model.load_state_dict(best_model_state)
            logger.info("Restored best model weights")

        # Save final model
        if save_path:
            self.save_model(save_path)

        self.callbacks.on_train_end(history)

        logger.info("Training completed!")
        return history

    def save_model(self, filepath: str):
        """Save trained model."""
        metadata = {
            'cancer_types': self.cancer_types,
            'training_config': self.config['training'],
            'best_metrics': {},  # Would be filled from validation
            'timestamp': datetime.now().isoformat()
        }

        self.model.save_model(filepath, metadata)

    def get_training_summary(self) -> Dict:
        """Get training summary and statistics."""
        return {
            'device': str(self.device),
            'cancer_types': self.cancer_types,
            'batch_size': self.batch_size,
            'epochs_trained': len(self.history.get('epochs', [])) if hasattr(self, 'history') else 0,
            'final_learning_rate': self.optimizer.param_groups[0]['lr'],
            'model_info': self.model.get_model_info()
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in medical imaging.

    Formula: FL(p_t) = -α(1-p_t)^γ * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare classes
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Model logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        probs = torch.softmax(inputs, dim=1)

        # Get probability of correct class
        targets_one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
        pt = (probs * targets_one_hot).sum(dim=1)

        # Compute focal loss
        log_pt = torch.log(pt + 1e-8)
        loss = -self.alpha * (1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_data_loaders(cancer_types: List[str],
                       batch_size: int = 32,
                       config_path: str = "config.yaml") -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        cancer_types: Cancer types to include
        batch_size: Batch size for data loading
        config_path: Path to configuration file

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create multi-cancer dataset
    train_dataset = MultiCancerDataset(
        data_path=str(Path("data/processed")),
        cancer_types=cancer_types,
        split='train',
        transform=create_augmentation_pipeline('basic', 'general', config_path),
        config_path=config_path,
        augment=True
    )

    val_dataset = MultiCancerDataset(
        data_path=str(Path("data/processed")),
        cancer_types=cancer_types,
        split='val',
        transform=create_augmentation_pipeline('validation', 'general', config_path),
        config_path=config_path,
        augment=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


def train_model(cancer_types: List[str] = None,
               config_path: str = "config.yaml",
               save_path: str = None) -> Dict[str, List[float]]:
    """
    Main training function for multi-cancer detection.

    Args:
        cancer_types: List of cancer types to train (None for all configured)
        config_path: Path to configuration file
        save_path: Path to save trained model

    Returns:
        Training history
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if cancer_types is None:
        cancer_types = [cancer['name'] for cancer in config['cancer_types']]

    logger.info(f"Training multi-cancer model for: {cancer_types}")

    # Create model
    model = create_multi_cancer_model(cancer_types, config_path=config_path)

    # Create trainer
    trainer = TransferLearningTrainer(model, cancer_types, config_path)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(cancer_types,
                                                  trainer.batch_size,
                                                  config_path)

    # Set default save path
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"models/multi_cancer_{timestamp}.pth"

    # Train model
    history = trainer.train(train_loader, val_loader, save_path)

    # Print final results
    final_metrics = {k: v[-1] for k, v in history.items() if k != 'epochs'}
    logger.info("Final training metrics:")
    for metric, value in final_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Cancer AI Detection Model")
    parser.add_argument("--cancer_types", nargs="+", default=["lung", "breast"],
                       help="Cancer types to train")
    parser.add_argument("--config", default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--save_path", default=None,
                       help="Path to save trained model")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs")

    args = parser.parse_args()

    # Override config if specified
    if args.batch_size or args.epochs:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
        if args.epochs:
            config['training']['num_epochs'] = args.epochs

        # Save temporary config
        temp_config = "temp_config.yaml"
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        args.config = temp_config

    # Train model
    history = train_model(args.cancer_types, args.config, args.save_path)

    # Save training history
    history_path = args.save_path.replace('.pth', '_history.json') if args.save_path else 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training history saved to {history_path}")