# Inference Guide for Multi-Cancer AI Detection Platform

## Overview

This guide covers deployment and usage of trained multi-cancer AI models for clinical inference, including single-patient predictions, batch processing, and integration with clinical workflows.

## Prerequisites

### System Requirements
- Python 3.9+ with PyTorch
- 8GB+ RAM for inference
- GPU recommended for real-time performance
- Network access for cloud deployment (optional)

### Model Requirements
- Trained model checkpoint (`.pth` file)
- Configuration file (`config.yaml`)
- Preprocessed data pipeline (for validation)

## Quick Start Inference

### 1. Load Model
```python
from inference.predictor import CancerPredictor

# Initialize predictor
predictor = CancerPredictor(
    model_path="models/multi_cancer_model.pth",
    config_path="config.yaml"
)

print("Model loaded successfully!")
```

### 2. Single Image Prediction
```python
# Predict on a single image
result = predictor.predict_single_image(
    image_path="path/to/ct_scan.png",
    cancer_type="lung",
    generate_explanation=True
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Risk Level: {result['risk_level']}")
```

### 3. Batch Processing
```python
# Process multiple images
image_paths = ["scan1.png", "scan2.png", "scan3.png"]
results = predictor.predict_batch(
    image_paths=image_paths,
    cancer_type="lung",
    generate_explanations=True
)

# Get batch statistics
stats = predictor.get_prediction_statistics(results)
print(f"Processed {stats['valid_predictions']} images successfully")
```

## Detailed Inference Workflow

### Single Patient Prediction

#### 1. Image Preprocessing
```python
from data_pipeline.preprocessing import MedicalImagePreprocessor

# Initialize preprocessor
preprocessor = MedicalImagePreprocessor()

# Preprocess image
processed_image = preprocessor.preprocess_image(
    original_image,  # numpy array or PIL Image
    modality="CT",   # or "MRI", "mammogram", etc.
    augment=False    # No augmentation for inference
)
```

#### 2. Model Prediction
```python
import torch

# Load and preprocess image
image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0).float()
image_tensor = image_tensor.to(device)

# Make prediction
with torch.no_grad():
    outputs = model(image_tensor)

    # Get prediction results
    prediction = outputs['prediction'].item()
    confidence = outputs['probabilities'][0][prediction].item()
    uncertainty = outputs['uncertainty'].item()
```

#### 3. Clinical Interpretation
```python
# Add clinical context
clinical_result = predictor._add_clinical_interpretation({
    'prediction': classes[prediction],
    'confidence': confidence,
    'uncertainty': uncertainty,
    'cancer_type': cancer_type
}, cancer_type)

# Generate explanation
explanation = predictor._generate_prediction_explanation(
    image_tensor, original_image, clinical_result, cancer_type
)
```

### Batch Processing Pipeline

#### 1. Setup Batch Processor
```python
from inference.inference_pipeline import InferencePipeline

# Initialize pipeline
pipeline = InferencePipeline({
    'lung': 'models/lung_model.pth',
    'breast': 'models/breast_model.pth'
})

print("Inference pipeline ready")
```

#### 2. Prepare Batch Data
```python
# Prepare batch of clinical images
batch_data = [
    {
        'image_path': 'patient_001_ct.png',
        'cancer_type': 'lung',
        'patient_id': 'PAT001',
        'study_date': '2024-01-15',
        'modality': 'CT',
        'clinical_notes': 'Follow-up for lung nodule'
    },
    {
        'image_path': 'patient_002_mammo.png',
        'cancer_type': 'breast',
        'patient_id': 'PAT002',
        'study_date': '2024-01-16',
        'modality': 'mammogram',
        'bi_rads_score': 3
    }
    # ... more patients
]
```

#### 3. Process Clinical Batch
```python
# Process batch with quality control
results = pipeline.process_clinical_batch(
    image_batch=batch_data,
    priority_processing=False  # Set True for urgent cases
)

print(f"Batch processed: {results['status']}")
print(f"Processing time: {results['processing_time']:.2f}s")
print(f"Quality score: {results['quality_metrics']['batch_quality_score']:.2f}")
```

#### 4. Handle Results
```python
# Process results
for result in results['results']:
    if result['status'] == 'success':
        print(f"Patient {result['patient_id']}: {result['prediction']} "
              f"(Confidence: {result['confidence']:.1%})")

        # Check for quality issues
        if result.get('requires_review'):
            print(f"⚠️ Review required: {result['quality_flags']}")

        # Get clinical workflow
        workflow = result.get('clinical_workflow', {})
        if workflow.get('priority') == 'high':
            print("🚨 HIGH PRIORITY - Immediate attention required")
    else:
        print(f"❌ Failed to process patient {result['patient_id']}: {result['error']}")
```

## Explainable AI Integration

### Generate Visual Explanations

```python
from xai.grad_cam import XAIInterpreter

# Initialize XAI interpreter
interpreter = XAIInterpreter(model, target_layer="layer4")

# Generate comprehensive explanation
explanations = interpreter.explain_prediction(
    input_tensor,
    original_image,
    cancer_type="lung",
    method="gradcam"
)

# Access different explanation types
heatmap = explanations['heatmap']
overlay = explanations['overlay']
saliency = explanations['saliency']

# Analyze attention patterns
attention_info = interpreter.get_attention_regions(heatmap, threshold=0.5)
print(f"Found {attention_info['num_regions']} suspicious regions")
```

### Textual Explanations

```python
from xai.explanations import ExplanationGenerator

# Initialize explanation generator
explainer = ExplanationGenerator()

# Generate clinical explanation
text_explanation = explainer.generate_explanation(
    prediction_result=result,
    attention_analysis=attention_info,
    cancer_type="lung"
)

print("Clinical Explanation:")
print(text_explanation)
```

### Save Explanations for Review

```python
from xai.grad_cam import save_explanation_visualization

# Save visualizations
save_explanation_visualization(
    explanations={
        'heatmap': heatmap,
        'overlay': overlay,
        'saliency': saliency
    },
    save_path="clinical_reports/",
    filename_prefix=f"patient_{patient_id}"
)

# Export structured report
explainer.export_explanation_report(
    explanation={'textual_explanation': text_explanation},
    prediction=result,
    save_path="clinical_reports/",
    filename="clinical_report.txt"
)
```

## Clinical Integration

### EHR Integration Pattern

```python
def integrate_with_ehr(prediction_result, patient_id, encounter_id):
    """Integrate AI results with Electronic Health Record."""

    # Prepare structured data for EHR
    ehr_data = {
        'patient_id': patient_id,
        'encounter_id': encounter_id,
        'ai_model_version': prediction_result.get('model_version'),
        'cancer_type': prediction_result.get('cancer_type'),
        'prediction': prediction_result.get('prediction'),
        'confidence_score': prediction_result.get('confidence'),
        'risk_level': prediction_result.get('risk_level'),
        'clinical_significance': prediction_result.get('clinical_significance'),
        'requires_followup': prediction_result.get('requires_followup'),
        'recommended_action': prediction_result.get('clinical_workflow', {}).get('recommended_action'),
        'processing_timestamp': prediction_result.get('processed_at'),
        'ai_disclaimer': "AI assistance only - clinical correlation required"
    }

    # Send to EHR system (implementation depends on EHR API)
    # ehr_api.submit_ai_results(ehr_data)

    return ehr_data
```

### Clinical Workflow Integration

```python
def trigger_clinical_workflow(prediction_result):
    """Trigger appropriate clinical workflows based on AI results."""

    workflow_actions = []

    risk_level = prediction_result.get('risk_level')
    cancer_type = prediction_result.get('cancer_type')

    if risk_level == 'HIGH':
        workflow_actions.extend([
            'urgent_specialist_consultation',
            'expedited_biopsy_scheduling',
            'multidisciplinary_tumor_board_review',
            'patient_notification_high_priority'
        ])

    elif risk_level == 'MEDIUM':
        workflow_actions.extend([
            'routine_followup_scheduling',
            'additional_imaging_recommendation',
            'primary_care_followup'
        ])

    # Cancer-specific workflows
    if cancer_type == 'lung' and risk_level in ['HIGH', 'MEDIUM']:
        workflow_actions.append('pulmonary_function_testing')
        workflow_actions.append('smoking_cessation_counseling')

    elif cancer_type == 'breast' and risk_level in ['HIGH', 'MEDIUM']:
        workflow_actions.append('genetic_counseling_referral')
        workflow_actions.append('breast_cancer_risk_assessment')

    return workflow_actions
```

### Quality Assurance

```python
def quality_assurance_check(prediction_result):
    """Perform quality assurance on AI predictions."""

    qa_issues = []

    # Check confidence thresholds
    confidence = prediction_result.get('confidence', 0)
    if confidence < 0.5:
        qa_issues.append("Low confidence prediction - requires secondary review")

    # Check for unusual patterns
    uncertainty = prediction_result.get('uncertainty', 0)
    if uncertainty > 0.8:
        qa_issues.append("High uncertainty - consider additional imaging")

    # Check prediction consistency
    if prediction_result.get('prediction') == 'malignant' and confidence < 0.6:
        qa_issues.append("Malignant prediction with low confidence - discordant")

    # Clinical context validation
    clinical_notes = prediction_result.get('clinical_notes', '').lower()
    prediction = prediction_result.get('prediction', '')

    if 'benign' in clinical_notes and prediction == 'malignant':
        qa_issues.append("Prediction conflicts with clinical impression")

    return qa_issues
```

## Performance Optimization

### Real-time Inference

```python
# Optimize for speed
torch.backends.cudnn.benchmark = True
model.eval()

# Use half precision if supported
if torch.cuda.is_available():
    model.half()  # Convert to FP16
    input_tensor = input_tensor.half()

# Batch processing for multiple images
batch_size = 8
with torch.no_grad():
    predictions = model(batch_input)  # Process 8 images at once
```

### Memory Management

```python
# Clear GPU memory between predictions
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Process large batches with memory management
def process_large_batch(image_paths, batch_size=4):
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]

        # Process batch
        batch_results = predictor.predict_batch(batch_paths, cancer_type)

        # Clear memory
        clear_memory()

        results.extend(batch_results)

    return results
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Use multiple CPU cores for I/O bound tasks
num_workers = multiprocessing.cpu_count()

def parallel_inference(image_paths, cancer_type):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit prediction tasks
        futures = [
            executor.submit(predictor.predict_single_image, path, cancer_type)
            for path in image_paths
        ]

        # Collect results
        results = [future.result() for future in futures]

    return results
```

## Deployment Architectures

### Local Deployment

```python
# Simple local server
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.json
    image_path = data['image_path']
    cancer_type = data['cancer_type']

    result = predictor.predict_single_image(image_path, cancer_type)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Docker Deployment

```dockerfile
# Dockerfile for inference service
FROM pytorch/pytorch:2.0-cuda11.8-runtime

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 5000

CMD ["python", "inference_service.py"]
```

### Cloud Deployment

```python
# AWS Lambda example
import json
import boto3
import base64
from PIL import Image
import io

def lambda_handler(event, context):
    # Decode base64 image
    image_data = base64.b64decode(event['image'])
    image = Image.open(io.BytesIO(image_data))

    # Save temporarily
    temp_path = '/tmp/temp_image.png'
    image.save(temp_path)

    # Make prediction
    result = predictor.predict_single_image(temp_path, event['cancer_type'])

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

## Monitoring and Maintenance

### Performance Monitoring

```python
class InferenceMonitor:
    def __init__(self):
        self.metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'average_confidence': 0,
            'error_rate': 0,
            'average_processing_time': 0
        }

    def update_metrics(self, result, processing_time):
        self.metrics['total_predictions'] += 1

        if result.get('status') == 'success':
            self.metrics['successful_predictions'] += 1
            self.metrics['average_confidence'] = (
                self.metrics['average_confidence'] +
                result.get('confidence', 0)
            ) / 2

        self.metrics['error_rate'] = (
            1 - self.metrics['successful_predictions'] / self.metrics['total_predictions']
        )

        self.metrics['average_processing_time'] = (
            self.metrics['average_processing_time'] + processing_time
        ) / 2

    def get_health_status(self):
        error_rate = self.metrics['error_rate']
        avg_time = self.metrics['average_processing_time']

        if error_rate > 0.1 or avg_time > 30:
            return 'CRITICAL'
        elif error_rate > 0.05 or avg_time > 10:
            return 'WARNING'
        else:
            return 'HEALTHY'
```

### Model Updates

```python
def check_model_drift(current_performance, baseline_metrics):
    """Check for model performance drift."""

    drift_threshold = 0.05  # 5% change

    metrics_to_check = ['sensitivity', 'specificity', 'auc']

    for metric in metrics_to_check:
        current_value = current_performance.get(metric, 0)
        baseline_value = baseline_metrics.get(metric, 0)

        if abs(current_value - baseline_value) > drift_threshold:
            print(f"⚠️ Drift detected in {metric}: {baseline_value:.3f} → {current_value:.3f}")
            return True

    return False

# Regular validation
def validate_model_performance():
    """Validate model on held-out test set."""
    test_metrics = predictor.validate_prediction_quality(
        validation_images=test_image_paths,
        validation_labels=test_labels,
        cancer_type='lung'
    )

    # Check for drift
    if check_model_drift(test_metrics, baseline_metrics):
        print("Model retraining recommended")
        # Trigger retraining pipeline

    return test_metrics
```

## Troubleshooting

### Common Issues

#### 1. Memory Errors
```python
# Reduce batch size
predictor.batch_size = 1

# Process images one at a time
for image_path in image_paths:
    result = predictor.predict_single_image(image_path, cancer_type)
    clear_memory()
```

#### 2. Slow Inference
```python
# Enable optimization
torch.backends.cudnn.benchmark = True
model.eval()

# Use TensorRT or ONNX for faster inference
# (Requires additional setup)
```

#### 3. Quality Issues
```python
# Check image preprocessing
preprocessed = preprocessor.preprocess_image(image, modality)
print(f"Image shape: {preprocessed.shape}")
print(f"Value range: {preprocessed.min():.3f} - {preprocessed.max():.3f}")

# Validate model inputs
print(f"Model expects: {model.input_shape}")
```

#### 4. Clinical Integration Issues
```python
# Validate EHR data format
required_fields = ['patient_id', 'prediction', 'confidence', 'risk_level']
missing_fields = [f for f in required_fields if f not in result]
if missing_fields:
    print(f"Missing required fields for EHR: {missing_fields}")
```

## Security and Compliance

### HIPAA Compliance

```python
def ensure_phi_protection(result):
    """Ensure PHI protection in results."""

    # Remove or mask sensitive information
    protected_result = result.copy()

    # Remove patient identifiers if not needed
    phi_fields = ['patient_name', 'date_of_birth', 'address']
    for field in phi_fields:
        protected_result.pop(field, None)

    # Add privacy notice
    protected_result['privacy_notice'] = "Patient identifiers removed for privacy"

    return protected_result
```

### Audit Logging

```python
import logging
import json
from datetime import datetime

def log_prediction_audit(result, user_id, session_id):
    """Log prediction for audit purposes."""

    audit_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'session_id': session_id,
        'prediction_id': result.get('prediction_id'),
        'cancer_type': result.get('cancer_type'),
        'confidence': result.get('confidence'),
        'risk_level': result.get('risk_level'),
        'model_version': result.get('model_version'),
        'processing_time': result.get('processing_time')
    }

    # Log to secure audit system
    logging.info(f"AUDIT: {json.dumps(audit_entry)}")

    # Could also send to SIEM system
    # siem_client.log_event('ai_prediction', audit_entry)
```

## Next Steps

After setting up inference:

1. **Clinical Validation**: Test on real clinical workflows
2. **User Training**: Train clinical staff on AI interpretation
3. **Integration Testing**: Validate with existing systems
4. **Monitoring Setup**: Implement continuous performance monitoring
5. **Documentation**: Create clinical user guides and protocols

For training instructions, see `docs/TRAINING.md`.
For ethics and safety guidelines, see `docs/ETHICS.md`.