# Multi-Cancer AI Early Detection Platform

## Overview

A production-grade, clinical-ready AI system for early detection of 8 major cancer types using transfer learning with shared feature extractors. This system provides visual and textual explainability for every prediction, prioritizing sensitivity and clinical interpretability.

## 🎯 Key Features

- **8 Cancer Types**: Brain, Lung, Breast, Skin, Cervical, Colorectal, Prostate, Liver
- **Transfer Learning**: Shared ResNet-50 backbone with cancer-specific classification heads
- **Explainable AI**: Grad-CAM heatmaps, saliency maps, and textual explanations
- **Clinical Metrics**: Sensitivity, specificity, ROC-AUC, false negative rate analysis
- **Medical-Grade Evaluation**: Beyond accuracy - focuses on clinical utility
- **Modular Architecture**: Production-ready, extensible, and reproducible

## 📋 System Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ storage for datasets and models

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd multi-cancer-ai

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Download datasets (example for lung cancer)
python -c "from data_pipeline.loaders import LungCancerLoader; loader = LungCancerLoader(); loader.download_data()"

# Preprocess data
python data_pipeline/preprocessing.py
```

### 3. Training

```bash
# Train model
python training/train.py --config config.yaml --cancer_types lung breast
```

### 4. Inference

```bash
# Run inference on single image
python inference/inference.py --image_path path/to/image.jpg --cancer_type lung
```

### 5. Launch Demo

```bash
# Start Streamlit interface
streamlit run app/streamlit_app.py
```

## 🏗️ Architecture

```
Shared Feature Extractor (ResNet-50)
        ↓
    Frozen Layers
        ↓
    Progressive Unfreezing
        ↓
[Head 1]  [Head 2]  ...  [Head 8]
Brain     Lung           Liver
(Binary Classification per Cancer)
```

## 📊 Supported Cancer Types

| Cancer Type | Modality | Key Features | Dataset |
|-------------|----------|--------------|---------|
| **Lung** | CT/X-ray | Nodules, spiculated edges | LIDC-IDRI |
| **Breast** | Mammogram | Microcalcifications, masses | CBIS-DDSM |
| **Brain** | MRI/CT | Tissue intensity, edema | BraTS |
| **Skin** | Dermoscopy | Asymmetry, borders | ISIC |
| **Cervical** | Pap Smear | Nuclear enlargement | SIPaKMeD |
| **Colorectal** | Histology | Glandular distortion | NCT-CRC-HE |
| **Prostate** | Histology | Gland architecture | Public datasets |
| **Liver** | CT/MRI | Enhancement patterns | Public datasets |

## 🧠 Explainable AI (XAI)

For every prediction, the system generates:

1. **Grad-CAM Heatmap**: Visualizes discriminative regions
2. **Saliency Map**: Shows pixel-level importance
3. **Text Explanation**: Human-readable clinical interpretation
4. **Confidence Score**: With uncertainty quantification

Example Output:
```
Prediction: Lung Cancer (Malignant Nodule)
Confidence: 94%
Risk Level: HIGH
Explanation: "Model identified a 12mm spiculated nodule with irregular borders
and peripheral ground-glass opacity in the right lower lobe, consistent with
early-stage malignancy."
```

## 📈 Clinical Metrics

The system evaluates using medical-grade metrics:

- **Sensitivity (Recall)**: TP/(TP+FN) - Detects all true cancers
- **Specificity**: TN/(TN+FP) - Minimizes false alarms
- **ROC-AUC**: Discrimination ability
- **F1-Score**: Balance of precision and recall
- **False Negative Rate**: Clinical risk assessment

## 🛡️ Safety & Ethics

### ⚠️ Important Disclaimers

**This AI system is ASSISTIVE ONLY. All predictions must be reviewed by qualified medical professionals.**

- No autonomous medical decisions
- Flags risk levels but does not recommend treatment
- Documents dataset biases and limitations
- Implements uncertainty quantification

### HIPAA/GDPR Compliance

- No patient PHI in codebase
- Assumes encrypted data pipelines
- Logs predictions without identifiable information
- Implements access controls for clinical deployment

## 📁 Project Structure

```
multi-cancer-ai/
├── README.md                    # This file
├── requirements.txt            # Dependencies
├── config.yaml                 # Configuration
├── data/                       # Data management
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Preprocessed data
│   └── splits/                 # Train/val/test splits
├── models/                     # Model architectures
│   ├── backbone.py             # Shared feature extractor
│   ├── heads.py                # Cancer-specific heads
│   └── multi_cancer_model.py   # Main model class
├── training/                   # Training pipeline
│   ├── train.py                # Main training script
│   ├── augmentation.py         # Data augmentation
│   └── callbacks.py            # Training callbacks
├── data_pipeline/              # Data processing
│   ├── dataset.py              # PyTorch datasets
│   ├── preprocessing.py        # Image preprocessing
│   └── loaders.py              # Cancer-specific loaders
├── xai/                        # Explainable AI
│   ├── grad_cam.py             # Grad-CAM implementation
│   ├── saliency.py             # Saliency maps
│   └── explanations.py         # Text explanations
├── evaluation/                 # Model evaluation
│   ├── metrics.py              # Clinical metrics
│   ├── clinical_analysis.py    # Medical interpretation
│   └── plots.py                # Visualization
├── inference/                  # Inference engine
│   ├── predictor.py            # Prediction logic
│   └── inference_pipeline.py   # Full pipeline
├── app/                        # User interfaces
│   └── streamlit_app.py        # Clinical demo
├── tests/                      # Unit tests
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_xai.py
└── docs/                       # Documentation
    ├── TRAINING.md
    ├── INFERENCE.md
    └── ETHICS.md
```

## 🔧 Configuration

All system parameters are configurable via `config.yaml`:

- Model architecture (backbone, heads, dropout)
- Training hyperparameters (learning rate, batch size, epochs)
- Data preprocessing (normalization, augmentation)
- XAI settings (layers, methods)
- Clinical thresholds and risk levels

## 🏥 Clinical Usage

### Single Patient Prediction

```python
from inference.predictor import CancerPredictor

predictor = CancerPredictor(model_path="models/lung_model.pth")
result = predictor.predict(image_path="patient_scan.jpg", cancer_type="lung")

print(f"Cancer Detected: {result['detected']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Risk Level: {result['risk_level']}")
print(f"Explanation: {result['explanation']}")
```

### Batch Processing

```python
# Process multiple patients
results = predictor.predict_batch(image_paths=patient_images, cancer_type="lung")
```

## 🔬 Research & Development

### Adding New Cancer Types

1. Add configuration to `config.yaml`
2. Implement cancer-specific loader in `data_pipeline/loaders.py`
3. Add classification head in `models/heads.py`
4. Update training pipeline if needed

### Custom Metrics

Extend `evaluation/metrics.py` with domain-specific metrics:

```python
def custom_clinical_metric(y_true, y_pred, y_prob):
    # Implement clinical utility metric
    pass
```

## 📚 Documentation

- **[Training Guide](docs/TRAINING.md)**: Detailed training procedures
- **[Inference Guide](docs/INFERENCE.md)**: Deployment and usage
- **[Ethics Guidelines](docs/ETHICS.md)**: Safety and bias considerations

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-cancer-type`
3. Add tests for new functionality
4. Ensure clinical validation of changes
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚕️ Clinical Validation

This system is designed for research and clinical investigation. Before deployment:

1. Validate on local patient cohorts
2. Compare with existing clinical workflows
3. Establish performance baselines
4. Train clinical staff on AI interpretation
5. Implement continuous monitoring and updates

## 📞 Support

For technical issues, research collaboration, or clinical deployment inquiries:

- Create GitHub issue for bugs
- Use Discussions for questions
- Email for clinical partnerships

---

**⚠️ MEDICAL DISCLAIMER**: This software is for research purposes only and should not be used for clinical decision-making without proper validation and regulatory approval.