"""
Training Callbacks for Model Training

Provides callbacks for:
- Model checkpointing
- Early stopping
- Learning rate scheduling
- Training monitoring
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
import numpy as np
import yaml
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingCallbacks:
    """
    Collection of training callbacks for monitoring and control.

    Manages multiple callbacks and coordinates their execution.
    """

    def __init__(self, callbacks: List['Callback'] = None):
        """Initialize with list of callbacks."""
        self.callbacks = callbacks or []

    def add_callback(self, callback: 'Callback'):
        """Add a callback to the collection."""
        self.callbacks.append(callback)

    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        """Called at the beginning of each epoch."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Called at the end of each epoch."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: Dict = None):
        """Called at the beginning of each batch."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Dict = None):
        """Called at the end of each batch."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs: Dict = None):
        """Called at the beginning of training."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Dict = None):
        """Called at the end of training."""
        for callback in self.callbacks:
            callback.on_train_end(logs)


class Callback:
    """Base callback class."""

    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        pass

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        pass

    def on_batch_begin(self, batch: int, logs: Dict = None):
        pass

    def on_batch_end(self, batch: int, logs: Dict = None):
        pass

    def on_train_begin(self, logs: Dict = None):
        pass

    def on_train_end(self, logs: Dict = None):
        pass


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    Saves models based on monitored metric or periodically.
    """

    def __init__(self,
                 filepath: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_freq: str = 'epoch',
                 verbose: bool = True):
        """
        Initialize model checkpoint callback.

        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max' for the monitored metric
            save_best_only: Save only when metric improves
            save_freq: 'epoch' or integer (save every N epochs)
            verbose: Print checkpoint messages
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.verbose = verbose

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.epoch_count = 0

        # Create directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Check if model should be saved."""
        logs = logs or {}
        self.epoch_count += 1

        should_save = False

        if self.save_freq == 'epoch':
            should_save = True
        elif isinstance(self.save_freq, int):
            should_save = (self.epoch_count % self.save_freq) == 0

        if should_save and self.save_best_only:
            current_value = logs.get(self.monitor)
            if current_value is not None:
                if self.mode == 'min' and current_value < self.best_value:
                    self.best_value = current_value
                    should_save = True
                elif self.mode == 'max' and current_value > self.best_value:
                    self.best_value = current_value
                    should_save = True
                else:
                    should_save = False

        if should_save:
            self._save_checkpoint(epoch, logs)

    def _save_checkpoint(self, epoch: int, logs: Dict):
        """Save model checkpoint."""
        checkpoint_path = self.filepath.parent / f"{self.filepath.stem}_epoch_{epoch:03d}{self.filepath.suffix}"

        # This would need access to the model - will be handled by trainer
        if self.verbose:
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            if self.monitor in logs:
                logger.info(f"Current {self.monitor}: {logs[self.monitor]:.4f}")


class EarlyStopping(Callback):
    """
    Stop training when monitored metric stops improving.

    Implements patience-based early stopping.
    """

    def __init__(self,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 patience: int = 10,
                 min_delta: float = 0.0,
                 restore_best_weights: bool = False,
                 verbose: bool = True):
        """
        Initialize early stopping callback.

        Args:
            monitor: Metric to monitor
            mode: 'min' or 'max' for the monitored metric
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Restore model weights from best epoch
            verbose: Print early stopping messages
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Check if training should stop."""
        logs = logs or {}
        current_value = logs.get(self.monitor)

        if current_value is None:
            return

        # Check if current value is better than best
        if self.mode == 'min':
            is_better = current_value < (self.best_value - self.min_delta)
        else:
            is_better = current_value > (self.best_value + self.min_delta)

        if is_better:
            self.best_value = current_value
            self.wait_count = 0
            self.best_epoch = epoch

            # Save best weights if requested
            if self.restore_best_weights:
                # This would need access to the model - handled by trainer
                pass
        else:
            self.wait_count += 1

        # Check if patience is exceeded
        if self.wait_count >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                logger.info(f"Early stopping at epoch {epoch}. Best {self.monitor}: {self.best_value:.4f}")

    def should_stop(self) -> bool:
        """Return whether training should stop."""
        return self.stopped_epoch > 0


class LearningRateScheduler(Callback):
    """
    Learning rate scheduling callback.

    Supports various LR scheduling strategies.
    """

    def __init__(self,
                 scheduler_type: str = 'cosine',
                 **scheduler_kwargs):
        """
        Initialize LR scheduler callback.

        Args:
            scheduler_type: Type of scheduler ('cosine', 'step', 'plateau', 'exponential')
            **scheduler_kwargs: Additional arguments for scheduler
        """
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler = None

    def set_scheduler(self, scheduler):
        """Set the PyTorch scheduler object."""
        self.scheduler = scheduler

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Update learning rate."""
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Plateau scheduler needs validation metric
                val_loss = logs.get('val_loss')
                if val_loss is not None:
                    self.scheduler.step(val_loss)
            else:
                self.scheduler.step()


class TrainingLogger(Callback):
    """
    Log training progress and metrics.

    Saves training logs to file and provides formatted output.
    """

    def __init__(self,
                 log_dir: str = 'logs',
                 log_filename: str = None,
                 verbose: bool = True):
        """
        Initialize training logger.

        Args:
            log_dir: Directory to save logs
            log_filename: Log filename (auto-generated if None)
            verbose: Print training progress
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if log_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'training_{timestamp}.log'

        self.log_path = self.log_dir / log_filename
        self.verbose = verbose

        # Initialize log data
        self.history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def on_train_begin(self, logs: Dict = None):
        """Initialize training logging."""
        logger.info("Training started")
        logger.info(f"Logs will be saved to: {self.log_path}")

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Log epoch results."""
        logs = logs or {}

        # Update history
        self.history['epochs'].append(epoch)
        for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            if key in logs:
                self.history[key].append(logs[key])

        # Format log message
        message = f"Epoch {epoch:3d}: "
        metrics = []
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                metrics.append(f"{key}={value:.4f}")
        message += ", ".join(metrics)

        # Print and save
        if self.verbose:
            print(message)

        with open(self.log_path, 'a') as f:
            f.write(message + '\n')

    def on_train_end(self, logs: Dict = None):
        """Save final training summary."""
        logger.info("Training completed")

        # Save history to JSON
        history_path = self.log_path.with_suffix('.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Training history saved to: {history_path}")


class ProgressTracker(Callback):
    """
    Track training progress and estimate completion time.

    Provides progress bars and time estimates.
    """

    def __init__(self, total_epochs: int, verbose: bool = True):
        """Initialize progress tracker."""
        self.total_epochs = total_epochs
        self.verbose = verbose
        self.start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, logs: Dict = None):
        """Start tracking training time."""
        from datetime import datetime
        self.start_time = datetime.now()
        if self.verbose:
            print(f"Training started at {self.start_time.strftime('%H:%M:%S')}")

    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        """Track epoch start time."""
        from datetime import datetime
        self.epoch_start_time = datetime.now()

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Display epoch progress and time estimates."""
        if not self.verbose:
            return

        from datetime import datetime
        epoch_end_time = datetime.now()

        # Calculate times
        epoch_duration = (epoch_end_time - self.epoch_start_time).total_seconds()
        elapsed_time = (epoch_end_time - self.start_time).total_seconds()

        # Estimate remaining time
        remaining_epochs = self.total_epochs - epoch - 1
        avg_epoch_time = elapsed_time / (epoch + 1)
        estimated_remaining = remaining_epochs * avg_epoch_time

        # Format progress message
        progress = (epoch + 1) / self.total_epochs * 100
        print(f"Epoch {epoch + 1:3d}/{self.total_epochs} "
              f"[{progress:5.1f}%] "
              f"Time: {epoch_duration:.1f}s "
              f"ETA: {estimated_remaining/3600:.1f}h")


def create_training_callbacks(config_path: str = "config.yaml") -> TrainingCallbacks:
    """
    Create standard training callbacks based on configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured TrainingCallbacks object
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    callbacks = TrainingCallbacks()

    # Add training logger
    callbacks.add_callback(TrainingLogger())

    # Add model checkpoint
    checkpoint_config = config.get('training', {}).get('checkpoint', {})
    if checkpoint_config.get('enabled', True):
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_config.get('filepath', 'models/checkpoint.pth'),
            monitor=checkpoint_config.get('monitor', 'val_auc'),
            mode=checkpoint_config.get('mode', 'max'),
            save_best_only=checkpoint_config.get('save_best_only', True)
        )
        callbacks.add_callback(checkpoint)

    # Add early stopping
    early_stop_config = config.get('training', {}).get('early_stopping', {})
    if early_stop_config.get('enabled', True):
        early_stopping = EarlyStopping(
            monitor=early_stop_config.get('monitor', 'val_auc'),
            mode=early_stop_config.get('mode', 'max'),
            patience=early_stop_config.get('patience', 10),
            min_delta=early_stop_config.get('min_delta', 0.001)
        )
        callbacks.add_callback(early_stopping)

    return callbacks