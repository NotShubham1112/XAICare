# Training Guide for Multi-Cancer AI Detection Platform

## Overview

This guide provides detailed instructions for training the multi-cancer AI detection system with clinical-grade performance and safety features.

## Prerequisites

### System Requirements
- Python 3.9+
- PyTorch 2.0+ with CUDA support
- 16GB+ RAM, 32GB+ recommended
- NVIDIA GPU with 8GB+ VRAM
- 100GB+ storage for datasets and models

### Dependencies
```bash
pip install -r requirements.txt
```

### Data Requirements
- Medical imaging datasets (DICOM, PNG, JPEG)
- Patient-wise train/validation/test splits
- Ground truth labels with clinical verification
- Dataset size: 1000+ images per cancer type for robust training

## Quick Start Training

### 1. Prepare Data
```bash
# Prepare datasets for lung and breast cancer
python -c "
from data_pipeline.loaders import prepare_all_cancer_data
data = prepare_all_cancer_data(['lung', 'breast'])
print('Data preparation completed')
"
```

### 2. Configure Training
Edit `config.yaml` for your specific requirements:
```yaml
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 1e-4
  progressive_unfreezing: true
  use_weighted_loss: true
  early_stopping_patience: 10

model:
  backbone: "resnet50"
  freeze_backbone: true
```

### 3. Train Model
```bash
# Train on lung and breast cancer
python training/train.py --cancer_types lung breast --config config.yaml

# Or specify custom save path
python training/train.py --cancer_types lung breast --save_path models/my_model.pth
```

### 4. Monitor Training
Training progress is logged to `logs/training_*.log` and TensorBoard-compatible metrics are available.

## Detailed Training Workflow

### Phase 1: Foundation Setup

#### 1.1 Data Preparation
```python
from data_pipeline.loaders import prepare_all_cancer_data

# Prepare data for multiple cancer types
cancer_types = ['lung', 'breast']
processed_data = prepare_all_cancer_data(cancer_types)

# Verify data splits
for cancer_type in cancer_types:
    print(f"{cancer_type}: {len(processed_data[cancer_type])} total images")
```

#### 1.2 Model Architecture Setup
```python
from models.multi_cancer_model import create_multi_cancer_model

# Create model with shared backbone
model = create_multi_cancer_model(
    cancer_types=['lung', 'breast'],
    backbone_name='resnet50',
    pretrained=True
)

print(f"Model created with {model.get_model_info()['total_parameters']} parameters")
```

### Phase 2: Transfer Learning Training

#### 2.1 Initial Training (Frozen Backbone)
```python
from training.train import TransferLearningTrainer

# Initialize trainer
trainer = TransferLearningTrainer(
    model=model,
    cancer_types=['lung', 'breast'],
    config_path='config.yaml'
)

# Create data loaders
from training.train import create_data_loaders
train_loader, val_loader = create_data_loaders(['lung', 'breast'])

# Train with frozen backbone
history = trainer.train(train_loader, val_loader, save_path='models/frozen_backbone.pth')
```

#### 2.2 Progressive Unfreezing
```python
# Unfreeze backbone layers progressively
trainer.model.unfreeze_backbone_layers([4])  # Unfreeze layer4

# Continue training with unfrozen layers
history_2 = trainer.train(train_loader, val_loader, save_path='models/unfrozen_layer4.pth')

# Unfreeze more layers
trainer.model.unfreeze_backbone_layers([5])  # Unfreeze layer5
history_3 = trainer.train(train_loader, val_loader, save_path='models/final_model.pth')
```

### Phase 3: Fine-tuning and Optimization

#### 3.1 Learning Rate Scheduling
```python
# Implement cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    trainer.optimizer, T_0=10, T_mult=2
)

# Training loop with scheduling
for epoch in range(trainer.num_epochs):
    # Training code...
    scheduler.step()
```

#### 3.2 Weighted Loss for Imbalanced Data
```python
# Calculate class weights from training data
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Get labels from training dataset
train_labels = []
for batch in train_loader:
    train_labels.extend(batch[1].numpy())

class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.FloatTensor(class_weights).to(trainer.device)

# Use weighted loss
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

#### 3.3 Focal Loss for Hard Examples
```python
from training.train import FocalLoss

# Use focal loss for imbalanced medical data
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# This automatically handles class imbalance by focusing on hard examples
```

## Advanced Training Techniques

### Data Augmentation Strategies

#### Medical-Image-Safe Augmentation
```python
from training.augmentation import create_augmentation_pipeline

# Strong augmentation for data-limited scenarios
strong_aug = create_augmentation_pipeline('strong', 'lung')

# Moderate augmentation for standard training
standard_aug = create_augmentation_pipeline('basic', 'lung')

# Validation transforms (no augmentation)
val_aug = create_augmentation_pipeline('validation', 'lung')
```

#### Modality-Specific Augmentation
```python
# CT-specific augmentation (windowing, noise)
ct_aug = create_augmentation_pipeline('basic', 'CT')

# Mammogram-specific augmentation (contrast, flipping)
mammo_aug = create_augmentation_pipeline('basic', 'mammogram')

# Histopathology augmentation (staining, rotation)
hist_aug = create_augmentation_pipeline('basic', 'histopathology')
```

### Cross-Validation Training

```python
from sklearn.model_selection import StratifiedKFold

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}/5")

    # Create fold-specific data loaders
    train_loader = create_fold_loader(train_idx)
    val_loader = create_fold_loader(val_idx)

    # Train model for this fold
    model = create_multi_cancer_model(['lung', 'breast'])
    trainer = TransferLearningTrainer(model, ['lung', 'breast'])
    history = trainer.train(train_loader, val_loader)

    fold_results.append(history)

# Aggregate results across folds
avg_metrics = aggregate_fold_results(fold_results)
```

### Hyperparameter Optimization

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # Create model with suggested parameters
    model = create_multi_cancer_model(['lung', 'breast'])
    # ... configure model with suggested parameters ...

    # Train and evaluate
    trainer = TransferLearningTrainer(model, ['lung', 'breast'])
    history = trainer.train(train_loader, val_loader)

    return history['val_auc'][-1]  # Optimize for AUC

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best parameters: {study.best_params}")
```

## Model Evaluation During Training

### Clinical Metrics Monitoring

```python
from evaluation.metrics import calculate_clinical_metrics

def evaluate_clinical_performance(model, val_loader, cancer_type):
    """Evaluate model using clinical metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, metadata in val_loader:
            outputs = model(images)
            probs = outputs['probabilities']

            # Assuming binary classification
            pos_probs = probs[:, 1].cpu().numpy()
            preds = (pos_probs > 0.5).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(pos_probs)

    # Calculate clinical metrics
    metrics = calculate_clinical_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )

    return metrics

# Monitor during training
for epoch in range(num_epochs):
    # Training loop...

    # Clinical evaluation
    clinical_metrics = evaluate_clinical_performance(model, val_loader, 'lung')
    print(f"Epoch {epoch}: Sensitivity={clinical_metrics['sensitivity']:.3f}, "
          f"Specificity={clinical_metrics['specificity']:.3f}, "
          f"AUC={clinical_metrics['auc']:.3f}")
```

### Early Stopping with Clinical Metrics

```python
from training.callbacks import EarlyStopping

# Stop training based on clinical metrics
early_stopping = EarlyStopping(
    monitor='val_auc',  # Use AUC for early stopping
    mode='max',
    patience=10,
    min_delta=0.01
)

for epoch in range(num_epochs):
    # Training...

    # Validation with clinical metrics
    val_metrics = evaluate_clinical_performance(model, val_loader, 'lung')

    # Check early stopping
    early_stopping.on_epoch_end(epoch, {'val_auc': val_metrics['auc']})
    if early_stopping.should_stop():
        print(f"Early stopping at epoch {epoch}")
        break
```

## Training Best Practices

### 1. Data Quality Control
- Verify all images load correctly
- Check label consistency
- Validate patient-wise splits (no data leakage)
- Ensure representative demographics

### 2. Training Stability
- Use gradient clipping to prevent exploding gradients
- Implement learning rate warmup
- Monitor for overfitting with validation metrics
- Use mixed precision training for speed

### 3. Clinical Validation
- Always evaluate on held-out test set
- Report confidence intervals for metrics
- Perform subgroup analysis (by demographics, image quality)
- Validate on external datasets when possible

### 4. Model Interpretability
- Train with explainability from the start
- Validate that explanations make clinical sense
- Monitor for unexpected model behavior

## Troubleshooting

### Common Issues

#### 1. Memory Issues
```python
# Reduce batch size
batch_size = 16

# Use gradient accumulation
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

# Use mixed precision
scaler = torch.cuda.amp.GradScaler()
```

#### 2. Overfitting
```python
# Increase regularization
model.dropout_rate = 0.5

# Add weight decay
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-4)

# Use data augmentation
augmentation = create_augmentation_pipeline('strong', cancer_type)
```

#### 3. Poor Convergence
```python
# Adjust learning rate
lr = 1e-4  # Try 1e-3, 1e-5, etc.

# Change optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5, factor=0.5
)
```

#### 4. Class Imbalance
```python
# Use weighted sampling
sampler = WeightedRandomSampler(weights, len(dataset))

# Use focal loss
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# Oversample minority class
# (Implement in data loading pipeline)
```

## Performance Optimization

### GPU Optimization
```python
# Enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# Use pinned memory
dataloader = DataLoader(dataset, pin_memory=True)

# Use distributed training for multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### Memory Optimization
```python
# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache periodically
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
```

## Deployment Preparation

### Model Export
```python
# Save model with metadata
metadata = {
    'cancer_types': ['lung', 'breast'],
    'training_date': datetime.now().isoformat(),
    'clinical_metrics': final_metrics,
    'intended_use': 'clinical_decision_support'
}

model.save_model('models/production_model.pth', metadata)
```

### Validation Checklist
- [ ] Sensitivity > 90% for screening use cases
- [ ] Specificity > 85% to minimize false alarms
- [ ] AUC > 0.90 for good discrimination
- [ ] Explainability validates clinically
- [ ] No significant bias across subgroups
- [ ] Performance on external validation set

## Next Steps

After training, proceed to:
1. **Inference Setup**: Deploy model for clinical use
2. **Clinical Validation**: Test on real clinical workflows
3. **Monitoring**: Set up continuous performance monitoring
4. **Updates**: Plan for model updates as new data becomes available

For deployment instructions, see `docs/INFERENCE.md`.