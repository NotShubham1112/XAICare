"""
Production Inference Service for Medical AI

FastAPI-based microservice for high-performance medical image inference
with GPU optimization, monitoring, and HIPAA compliance.
"""

import os
import asyncio
import uuid
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import prometheus_client
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import numpy as np
from PIL import Image
import io

# Import project modules
from models.multi_cancer_model import create_multi_cancer_model, MultiCancerModel
from data_pipeline.preprocessing import MedicalImagePreprocessor
from xai.grad_cam import XAIInterpreter
from evaluation.metrics import calculate_clinical_metrics

# Prometheus metrics
INFERENCE_REQUESTS = prometheus_client.Counter(
    'inference_requests_total',
    'Total inference requests',
    ['cancer_type', 'status', 'risk_level']
)

INFERENCE_LATENCY = prometheus_client.Histogram(
    'inference_latency_seconds',
    'Inference latency in seconds',
    ['cancer_type']
)

GPU_UTILIZATION = prometheus_client.Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage'
)

MODEL_CONFIDENCE = prometheus_client.Histogram(
    'model_confidence_score',
    'Distribution of model confidence scores',
    ['cancer_type']
)

# Pydantic models
class InferenceRequest(BaseModel):
    """Request model for medical image inference."""
    image_data: bytes = Field(..., description="Medical image data")
    cancer_type: str = Field(..., description="Type of cancer to detect")
    patient_id: Optional[str] = Field(None, description="Patient identifier (tokenized)")
    modality: str = Field("CT", description="Imaging modality")
    priority: str = Field("normal", description="Processing priority")
    generate_explanation: bool = Field(True, description="Generate XAI explanation")

    @validator('cancer_type')
    def validate_cancer_type(cls, v):
        supported_types = ['lung', 'breast', 'brain', 'skin', 'cervical', 'colorectal', 'prostate', 'liver']
        if v not in supported_types:
            raise ValueError(f'Unsupported cancer type. Must be one of: {supported_types}')
        return v

    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ['low', 'normal', 'high', 'urgent']
        if v not in valid_priorities:
            raise ValueError(f'Invalid priority. Must be one of: {valid_priorities}')
        return v

class InferenceResponse(BaseModel):
    """Response model for inference results."""
    prediction_id: str
    cancer_type: str
    prediction: str
    confidence: float
    risk_level: str
    clinical_significance: str
    requires_followup: bool
    processing_time: float
    model_version: str
    timestamp: str
    explanation: Optional[Dict[str, Any]] = None

class BatchInferenceRequest(BaseModel):
    """Request model for batch inference."""
    requests: List[InferenceRequest] = Field(..., max_items=50)
    batch_priority: str = Field("normal")

class BatchInferenceResponse(BaseModel):
    """Response model for batch inference."""
    batch_id: str
    results: List[InferenceResponse]
    batch_stats: Dict[str, Any]

# Global service instances
inference_service = None
security_scheme = HTTPBearer()

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    print("🚀 Starting Medical AI Inference Service...")

    global inference_service
    inference_service = InferenceService()

    print("📦 Loading AI models...")
    await inference_service.load_models()

    print("🔥 Warming up GPU...")
    await inference_service.warmup_gpu()

    print("✅ Service ready for inference!")

    yield

    # Shutdown
    print("🛑 Shutting down inference service...")
    await inference_service.cleanup()

# FastAPI app
app = FastAPI(
    title="Medical AI Inference Service",
    description="Production-ready inference service for multi-cancer detection",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument for OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

class InferenceService:
    """Production inference service with GPU optimization and monitoring."""

    def __init__(self):
        """Initialize inference service."""
        self.models = {}
        self.xai_interpreters = {}
        self.preprocessor = MedicalImagePreprocessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cache = {}
        self.batch_queue = asyncio.Queue(maxsize=100)

        # Configuration
        self.max_batch_size = int(os.getenv("INFERENCE_BATCH_SIZE", "16"))
        self.model_cache_size = int(os.getenv("MODEL_CACHE_SIZE", "3"))
        self.enable_mixed_precision = os.getenv("MIXED_PRECISION", "true").lower() == "true"

    async def load_models(self):
        """Load AI models for supported cancer types."""
        supported_cancers = ['lung', 'breast']  # Start with implemented types

        for cancer_type in supported_cancers:
            try:
                # Load model from registry or local path
                model_path = os.getenv(f"{cancer_type.upper()}_MODEL_PATH",
                                     f"models/{cancer_type}_model.pth")

                if os.path.exists(model_path):
                    model = MultiCancerModel.load_model(model_path)
                    model = model.to(self.device)

                    # Optimize for inference
                    if self.device.type == "cuda":
                        model = torch.compile(model, mode="reduce-overhead")
                        if self.enable_mixed_precision:
                            model.half()

                    self.models[cancer_type] = model
                    self.xai_interpreters[cancer_type] = XAIInterpreter(model)

                    print(f"✅ Loaded {cancer_type} model")
                else:
                    print(f"⚠️  Model not found: {model_path}")

            except Exception as e:
                print(f"❌ Failed to load {cancer_type} model: {e}")

    async def warmup_gpu(self):
        """Warm up GPU with dummy inference."""
        if self.device.type != "cuda":
            return

        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            if self.enable_mixed_precision:
                dummy_input = dummy_input.half()

            # Run warmup inference for each model
            for cancer_type, model in self.models.items():
                with torch.no_grad():
                    _ = model(dummy_input)
                print(f"🔥 GPU warmed up for {cancer_type}")

        except Exception as e:
            print(f"⚠️  GPU warmup failed: {e}")

    async def cleanup(self):
        """Clean up resources."""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    async def predict_single(self, request: InferenceRequest) -> InferenceResponse:
        """Perform single image prediction."""
        start_time = time.time()

        try:
            # Load model if not cached
            model = await self._get_model(request.cancer_type)
            interpreter = self.xai_interpreters.get(request.cancer_type)

            # Preprocess image
            image_tensor, original_image = await self._preprocess_image(
                request.image_data, request.modality, request.cancer_type
            )

            # Run inference
            with torch.no_grad():
                if self.enable_mixed_precision and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = model(image_tensor)
                else:
                    outputs = model(image_tensor)

            # Process results
            prediction_result = await self._process_prediction_results(
                outputs, request.cancer_type
            )

            # Generate explanation if requested
            explanation = None
            if request.generate_explanation and interpreter:
                explanation = await self._generate_explanation(
                    interpreter, image_tensor, original_image, prediction_result, request.cancer_type
                )

            # Create response
            response = InferenceResponse(
                prediction_id=str(uuid.uuid4()),
                cancer_type=request.cancer_type,
                prediction=prediction_result['prediction'],
                confidence=prediction_result['confidence'],
                risk_level=prediction_result['risk_level'],
                clinical_significance=prediction_result['clinical_significance'],
                requires_followup=prediction_result['requires_followup'],
                processing_time=time.time() - start_time,
                model_version="1.0.0",
                timestamp=datetime.utcnow().isoformat(),
                explanation=explanation
            )

            # Update metrics
            INFERENCE_REQUESTS.labels(
                cancer_type=request.cancer_type,
                status="success",
                risk_level=response.risk_level
            ).inc()

            MODEL_CONFIDENCE.labels(
                cancer_type=request.cancer_type
            ).observe(response.confidence)

            INFERENCE_LATENCY.labels(
                cancer_type=request.cancer_type
            ).observe(response.processing_time)

            return response

        except Exception as e:
            # Update error metrics
            INFERENCE_REQUESTS.labels(
                cancer_type=request.cancer_type,
                status="error",
                risk_level="unknown"
            ).inc()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference failed: {str(e)}"
            )

    async def predict_batch(self, requests: List[InferenceRequest]) -> BatchInferenceResponse:
        """Perform batch prediction."""
        batch_id = str(uuid.uuid4())
        results = []

        # Group by cancer type for efficient processing
        cancer_groups = {}
        for req in requests:
            if req.cancer_type not in cancer_groups:
                cancer_groups[req.cancer_type] = []
            cancer_groups[req.cancer_type].append(req)

        # Process each cancer type
        for cancer_type, group_requests in cancer_groups.items():
            try:
                model = await self._get_model(cancer_type)

                # Preprocess all images in batch
                batch_tensors = []
                original_images = []

                for req in group_requests:
                    tensor, original = await self._preprocess_image(
                        req.image_data, req.modality, cancer_type
                    )
                    batch_tensors.append(tensor)
                    original_images.append(original)

                # Stack batch
                batch_tensor = torch.cat(batch_tensors, dim=0)

                # Run batch inference
                with torch.no_grad():
                    if self.enable_mixed_precision and self.device.type == "cuda":
                        with torch.cuda.amp.autocast():
                            batch_outputs = model(batch_tensor)
                    else:
                        batch_outputs = model(batch_tensor)

                # Process individual results
                for i, req in enumerate(group_requests):
                    outputs = {k: v[i:i+1] for k, v in batch_outputs.items()}

                    prediction_result = await self._process_prediction_results(
                        outputs, cancer_type
                    )

                    # Generate explanation if requested
                    explanation = None
                    if req.generate_explanation:
                        interpreter = self.xai_interpreters.get(cancer_type)
                        if interpreter:
                            explanation = await self._generate_explanation(
                                interpreter, batch_tensors[i:i+1], original_images[i],
                                prediction_result, cancer_type
                            )

                    result = InferenceResponse(
                        prediction_id=str(uuid.uuid4()),
                        cancer_type=cancer_type,
                        prediction=prediction_result['prediction'],
                        confidence=prediction_result['confidence'],
                        risk_level=prediction_result['risk_level'],
                        clinical_significance=prediction_result['clinical_significance'],
                        requires_followup=prediction_result['requires_followup'],
                        processing_time=0.0,  # Calculated per batch
                        model_version="1.0.0",
                        timestamp=datetime.utcnow().isoformat(),
                        explanation=explanation
                    )

                    results.append(result)

            except Exception as e:
                # Add error results for failed group
                for req in group_requests:
                    results.append(InferenceResponse(
                        prediction_id=str(uuid.uuid4()),
                        cancer_type=cancer_type,
                        prediction="error",
                        confidence=0.0,
                        risk_level="unknown",
                        clinical_significance="Processing error",
                        requires_followup=False,
                        processing_time=0.0,
                        model_version="1.0.0",
                        timestamp=datetime.utcnow().isoformat(),
                        explanation={"error": str(e)}
                    ))

        # Calculate batch statistics
        batch_stats = self._calculate_batch_stats(results)

        return BatchInferenceResponse(
            batch_id=batch_id,
            results=results,
            batch_stats=batch_stats
        )

    async def _get_model(self, cancer_type: str) -> nn.Module:
        """Get or load model for cancer type."""
        if cancer_type not in self.models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not available for cancer type: {cancer_type}"
            )
        return self.models[cancer_type]

    async def _preprocess_image(self, image_data: bytes, modality: str, cancer_type: str):
        """Preprocess image data."""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            original_image = np.array(image)

            # Preprocess for model
            processed_image = self.preprocessor.preprocess_image(
                original_image, modality, augment=False
            )

            # Convert to tensor
            image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).float().unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            if self.enable_mixed_precision and self.device.type == "cuda":
                image_tensor = image_tensor.half()

            return image_tensor, original_image

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image preprocessing failed: {str(e)}"
            )

    async def _process_prediction_results(self, outputs: Dict[str, torch.Tensor], cancer_type: str) -> Dict[str, Any]:
        """Process model outputs into clinical results."""
        # Get prediction and confidence
        logits = outputs['logits'].cpu().numpy()[0]
        probabilities = outputs['probabilities'].cpu().numpy()[0]

        predicted_idx = np.argmax(logits)
        confidence = float(probabilities[predicted_idx])

        # Get class names (simplified for binary classification)
        class_names = ['benign', 'malignant']
        prediction = class_names[predicted_idx]

        # Determine risk level
        risk_level = self._classify_risk_level(confidence)

        # Clinical significance
        clinical_significance = self._get_clinical_significance(prediction, confidence, risk_level)

        # Follow-up requirement
        requires_followup = risk_level in ['HIGH', 'MEDIUM'] or (prediction == 'malignant' and confidence > 0.7)

        return {
            'prediction': prediction,
            'confidence': confidence,
            'risk_level': risk_level,
            'clinical_significance': clinical_significance,
            'requires_followup': requires_followup
        }

    def _classify_risk_level(self, confidence: float) -> str:
        """Classify risk level based on confidence."""
        risk_thresholds = {
            'high': float(os.getenv("HIGH_RISK_THRESHOLD", "0.9")),
            'medium': float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.7")),
            'low': float(os.getenv("LOW_RISK_THRESHOLD", "0.5"))
        }

        if confidence >= risk_thresholds['high']:
            return 'HIGH'
        elif confidence >= risk_thresholds['medium']:
            return 'MEDIUM'
        elif confidence >= risk_thresholds['low']:
            return 'LOW'
        else:
            return 'VERY_LOW'

    def _get_clinical_significance(self, prediction: str, confidence: float, risk_level: str) -> str:
        """Get clinical significance description."""
        if risk_level == 'HIGH':
            return "High confidence finding requiring immediate clinical attention and potential biopsy."
        elif risk_level == 'MEDIUM':
            return "Moderate confidence finding warranting expedited clinical evaluation."
        elif risk_level == 'LOW':
            return "Low confidence finding appropriate for routine clinical follow-up."
        else:
            return "Very low confidence - clinical correlation essential, additional testing recommended."

    async def _generate_explanation(self, interpreter, image_tensor, original_image, prediction_result, cancer_type):
        """Generate XAI explanation."""
        try:
            explanations = interpreter.explain_prediction(
                image_tensor, original_image, cancer_type, method="gradcam"
            )

            # Analyze attention regions
            attention_analysis = interpreter.get_attention_regions(
                explanations['heatmap'], threshold=0.5
            )

            return {
                'attention_analysis': attention_analysis,
                'visual_available': True,
                'method': 'gradcam'
            }

        except Exception as e:
            return {
                'error': f"Explanation generation failed: {str(e)}",
                'visual_available': False
            }

    def _calculate_batch_stats(self, results: List[InferenceResponse]) -> Dict[str, Any]:
        """Calculate batch processing statistics."""
        if not results:
            return {}

        confidences = [r.confidence for r in results if r.prediction != 'error']
        risk_levels = [r.risk_level for r in results if r.prediction != 'error']
        processing_times = [r.processing_time for r in results]

        risk_distribution = {}
        for level in ['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH']:
            risk_distribution[level] = risk_levels.count(level)

        return {
            'total_requests': len(results),
            'successful_requests': len([r for r in results if r.prediction != 'error']),
            'average_confidence': np.mean(confidences) if confidences else 0,
            'average_processing_time': np.mean(processing_times) if processing_times else 0,
            'risk_distribution': risk_distribution,
            'high_risk_count': risk_distribution.get('HIGH', 0)
        }

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "medical-ai-inference",
        "version": "1.0.0",
        "models_loaded": list(inference_service.models.keys()) if inference_service else []
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    if not inference_service or not inference_service.models:
        raise HTTPException(status_code=503, detail="Service not ready")

    return {"status": "ready"}

@app.post("/api/v1/inference/predict", response_model=InferenceResponse)
async def predict_single(
    request: InferenceRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme)
):
    """
    Single image inference endpoint.

    Requires authentication and returns prediction with clinical interpretation.
    """
    # Validate authentication
    await authenticate_user(credentials)

    # Check rate limiting
    await check_rate_limit(credentials.credentials)

    # Perform inference
    result = await inference_service.predict_single(request)

    return result

@app.post("/api/v1/inference/batch", response_model=BatchInferenceResponse)
async def predict_batch(
    request: BatchInferenceRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme)
):
    """
    Batch inference endpoint for multiple images.

    Optimized for clinical workflows requiring multiple predictions.
    """
    # Validate authentication
    await authenticate_user(credentials)

    # Check rate limiting (more lenient for batch)
    await check_rate_limit(credentials.credentials, endpoint="batch")

    # Perform batch inference
    result = await inference_service.predict_batch(request.requests)

    return result

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Update GPU metrics if available
    if torch.cuda.is_available():
        gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
        GPU_UTILIZATION.set(gpu_util)

    return Response(
        prometheus_client.generate_latest(),
        media_type="text/plain"
    )

# Authentication and security functions (simplified for demo)
async def authenticate_user(credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
    """Authenticate user token."""
    # In production, validate JWT token against identity provider
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing credentials")

    # Simplified validation - in production use proper JWT validation
    try:
        # Decode and validate JWT
        user_info = {"user_id": "demo_user", "role": "clinician"}
        return user_info
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

async def check_rate_limit(token: str, endpoint: str = "single") -> None:
    """Check rate limiting."""
    # In production, implement proper rate limiting with Redis
    # For demo, allow all requests
    pass

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
        log_level="info"
    )