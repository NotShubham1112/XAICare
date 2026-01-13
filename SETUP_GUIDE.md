# Multi-Cancer AI Platform - Setup Guide

## 🚀 Quick Start Setup

### Option 1: Docker Setup (Recommended for Production)
```bash
# Clone the repository
git clone <repository-url>
cd multi-cancer-ai

# Build and run with Docker
docker build -t medical-ai .
docker run -p 8501:8501 -p 8000:8000 medical-ai

# Access interfaces:
# - Clinical UI: http://localhost:8501
# - API: http://localhost:8000
```

### Option 2: Local Development Setup

#### Prerequisites
- Python 3.9-3.11 (avoid 3.12+ for compatibility)
- pip >= 20.0
- Git

#### 1. Environment Setup
```bash
# Create virtual environment
python -m venv medical-ai-env
source medical-ai-env/bin/activate  # On Windows: medical-ai-env\Scripts\activate

# Clone repository
git clone <repository-url>
cd multi-cancer-ai
```

#### 2. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements-dev.txt

# For production features (may require additional setup)
pip install sqlalchemy redis prometheus-client

# Note: Some packages may require system dependencies
# On Ubuntu/Debian:
# sudo apt-get install libpq-dev python3-dev
```

#### 3. Verify Installation
```bash
python -c "
import torch
from PIL import Image
import fastapi
print('✅ Core dependencies installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'FastAPI: {fastapi.__version__}')
"
```

#### 4. Initialize Data
```bash
# Prepare sample datasets
python -c "
from data_pipeline.loaders import prepare_all_cancer_data
data = prepare_all_cancer_data(['lung', 'breast'])
print('Sample data prepared')
"
```

#### 5. Start Development Server
```bash
# Start clinical interface
streamlit run app/streamlit_app.py

# In another terminal, start API server
python backend/services/inference_service/main.py
```

## 🐳 Docker Development Setup

### Development Container
```dockerfile
# Dockerfile.dev
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Expose ports
EXPOSE 8501 8000

# Default command
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run
```bash
# Build development image
docker build -f Dockerfile.dev -t medical-ai-dev .

# Run with volume mounting for development
docker run -p 8501:8501 -p 8000:8000 \
  -v $(pwd):/app \
  -e PYTHONPATH=/app \
  medical-ai-dev
```

## 🔧 Troubleshooting

### Common Issues

#### 1. NumPy Installation Issues (Windows)
```bash
# Try installing NumPy separately
pip uninstall numpy
pip install numpy --only-binary=all

# Or use conda
conda install numpy
```

#### 2. PyTorch CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU-only version if CUDA unavailable
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Database Connection Issues
```bash
# For development, use SQLite instead of PostgreSQL
# Comment out PostgreSQL dependencies in requirements

# Or use Docker for databases
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_PASSWORD=password \
  postgres:13
```

#### 4. Memory Issues
```bash
# Reduce batch size in configuration
# Use smaller model variants
# Enable gradient checkpointing
```

### Performance Optimization

#### GPU Setup
```bash
# Verify GPU setup
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Optimization
```python
# In config.yaml, adjust settings:
model:
  batch_size: 8  # Reduce from 32
  use_gradient_checkpointing: true

training:
  accumulation_steps: 4  # Gradient accumulation
```

## 📊 Testing Setup

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run basic tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Clinical Validation
```bash
# Test model loading
python -c "
from models.multi_cancer_model import create_multi_cancer_model
model = create_multi_cancer_model(['lung', 'breast'])
print('Model created successfully')
"
```

## 🚀 Production Deployment

### Using Kubernetes
```bash
# Apply Kubernetes manifests
kubectl apply -f backend/infrastructure/kubernetes/

# Check deployment status
kubectl get pods -n medical-ai
kubectl logs -f deployment/inference-service -n medical-ai
```

### Using Docker Compose (Staging)
```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Deploy application
docker-compose -f docker-compose.prod.yml up -d
```

### Environment Variables
```bash
# Copy and configure environment
cp backend/infrastructure/config/production.env .env
# Edit .env with your production values

# Load environment
export $(cat .env | xargs)
```

## 🔍 Monitoring Setup

### Local Monitoring
```bash
# Start Prometheus and Grafana
docker-compose -f docker-compose.monitoring.yml up -d

# Access dashboards:
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

### Health Checks
```bash
# API health check
curl http://localhost:8000/health

# Model health check
curl http://localhost:8000/api/v1/inference/health
```

## 📚 Next Steps

1. **Explore the Codebase**: Review the modular architecture
2. **Run Demos**: Try the Streamlit interface with sample data
3. **Customize Models**: Modify cancer types and model architectures
4. **Add Data**: Integrate your own medical imaging datasets
5. **Deploy**: Follow production deployment guide for clinical use

## 🆘 Getting Help

- **Documentation**: See `docs/` folder for detailed guides
- **Issues**: Check GitHub issues for common problems
- **Discussions**: Use GitHub discussions for questions
- **Clinical Integration**: Contact for HIPAA-compliant deployment guidance

---

**🎉 Your Multi-Cancer AI Platform is now ready for development and deployment!**