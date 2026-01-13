#!/usr/bin/env python3
"""
Medical AI Prometheus Exporter

Exports custom metrics for Medical AI platform monitoring:
- Clinical performance metrics
- Model drift detection
- Business metrics
- System health indicators
"""

import os
import time
import logging
from typing import Dict, List, Optional
import psycopg2
import redis
import aioredis
import asyncio
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalAIMetricsExporter:
    """Prometheus exporter for Medical AI platform metrics."""

    def __init__(self):
        """Initialize the metrics exporter."""
        self.config_path = "config.yaml"
        self.load_config()

        # Database connections
        self.db_conn = None
        self.redis_conn = None

        # Prometheus metrics
        self.setup_prometheus_metrics()

    def load_config(self):
        """Load configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults")
            self.config = {}

    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics definitions."""

        # Clinical Performance Metrics
        self.clinical_sensitivity = Gauge(
            'medical_ai_clinical_sensitivity',
            'Clinical sensitivity (recall) by cancer type',
            ['cancer_type', 'model_version']
        )

        self.clinical_specificity = Gauge(
            'medical_ai_clinical_specificity',
            'Clinical specificity by cancer type',
            ['cancer_type', 'model_version']
        )

        self.clinical_auc = Gauge(
            'medical_ai_clinical_auc',
            'Clinical AUC score by cancer type',
            ['cancer_type', 'model_version']
        )

        self.clinical_f1_score = Gauge(
            'medical_ai_clinical_f1_score',
            'Clinical F1 score by cancer type',
            ['cancer_type', 'model_version']
        )

        # Model Performance Metrics
        self.model_inference_latency = Histogram(
            'medical_ai_model_inference_latency_seconds',
            'Model inference latency in seconds',
            ['cancer_type', 'model_version'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float('inf'))
        )

        self.model_confidence_score = Histogram(
            'medical_ai_model_confidence_score',
            'Distribution of model confidence scores',
            ['cancer_type'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )

        # Business Metrics
        self.prediction_requests_total = Counter(
            'medical_ai_prediction_requests_total',
            'Total prediction requests by cancer type and status',
            ['cancer_type', 'status', 'risk_level']
        )

        self.batch_processing_time = Histogram(
            'medical_ai_batch_processing_time_seconds',
            'Batch processing time in seconds',
            ['batch_size']
        )

        # System Health Metrics
        self.active_models = Gauge(
            'medical_ai_active_models',
            'Number of active models by cancer type',
            ['cancer_type']
        )

        self.model_drift_score = Gauge(
            'medical_ai_model_drift_score',
            'Model drift score (0-1, higher = more drift)',
            ['cancer_type', 'model_version']
        )

        # Physician Agreement Metrics
        self.physician_agreement_rate = Gauge(
            'medical_ai_physician_agreement_rate',
            'Rate of physician agreement with AI predictions',
            ['cancer_type', 'time_period']
        )

        # Data Quality Metrics
        self.dicom_processing_success_rate = Gauge(
            'medical_ai_dicom_processing_success_rate',
            'DICOM processing success rate',
            ['processing_step']
        )

        # Resource Usage Metrics
        self.gpu_memory_usage = Gauge(
            'medical_ai_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['gpu_id', 'cancer_type']
        )

        self.cache_hit_rate = Gauge(
            'medical_ai_cache_hit_rate',
            'Model cache hit rate',
            ['cache_type']
        )

    async def connect_databases(self):
        """Connect to databases."""
        try:
            # PostgreSQL connection
            db_url = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/medicalai')
            self.db_conn = psycopg2.connect(db_url)

            # Redis connection
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_conn = redis.from_url(redis_url)

            logger.info("Database connections established")

        except Exception as e:
            logger.error(f"Database connection failed: {e}")

    async def collect_clinical_metrics(self):
        """Collect clinical performance metrics from database."""
        if not self.db_conn:
            return

        try:
            cursor = self.db_conn.cursor()

            # Get latest clinical metrics for each cancer type
            query = """
            SELECT
                cancer_type,
                model_version,
                sensitivity,
                specificity,
                auc,
                f1_score
            FROM clinical_metrics
            WHERE metric_date = CURRENT_DATE
            ORDER BY created_at DESC
            """

            cursor.execute(query)
            results = cursor.fetchall()

            for row in results:
                cancer_type, model_version, sensitivity, specificity, auc, f1 = row

                # Update Prometheus metrics
                self.clinical_sensitivity.labels(
                    cancer_type=cancer_type, model_version=model_version
                ).set(sensitivity or 0)

                self.clinical_specificity.labels(
                    cancer_type=cancer_type, model_version=model_version
                ).set(specificity or 0)

                self.clinical_auc.labels(
                    cancer_type=cancer_type, model_version=model_version
                ).set(auc or 0)

                self.clinical_f1_score.labels(
                    cancer_type=cancer_type, model_version=model_version
                ).set(f1 or 0)

        except Exception as e:
            logger.error(f"Failed to collect clinical metrics: {e}")

    async def collect_business_metrics(self):
        """Collect business metrics from database."""
        if not self.db_conn:
            return

        try:
            cursor = self.db_conn.cursor()

            # Get prediction counts by cancer type and risk level
            query = """
            SELECT
                cancer_type,
                risk_level,
                COUNT(*) as count
            FROM predictions
            WHERE created_at >= CURRENT_DATE - INTERVAL '1 day'
            GROUP BY cancer_type, risk_level
            """

            cursor.execute(query)
            results = cursor.fetchall()

            for cancer_type, risk_level, count in results:
                self.prediction_requests_total.labels(
                    cancer_type=cancer_type,
                    status='success',
                    risk_level=risk_level
                ).inc(count)

        except Exception as e:
            logger.error(f"Failed to collect business metrics: {e}")

    async def collect_system_health_metrics(self):
        """Collect system health metrics."""
        if not self.redis_conn:
            return

        try:
            # Get active model count from Redis
            active_models = self.redis_conn.hgetall('active_models') or {}

            for cancer_type, count in active_models.items():
                self.active_models.labels(cancer_type=cancer_type).set(int(count))

            # Get cache hit rate
            cache_stats = self.redis_conn.hgetall('cache_stats') or {}
            if 'hits' in cache_stats and 'misses' in cache_stats:
                hits = int(cache_stats['hits'])
                misses = int(cache_stats['misses'])
                total = hits + misses
                hit_rate = hits / total if total > 0 else 0

                self.cache_hit_rate.labels(cache_type='model').set(hit_rate)

        except Exception as e:
            logger.error(f"Failed to collect system health metrics: {e}")

    async def collect_model_drift_metrics(self):
        """Collect model drift detection metrics."""
        if not self.db_conn:
            return

        try:
            cursor = self.db_conn.cursor()

            # Calculate drift scores (simplified - compare recent vs baseline performance)
            query = """
            SELECT
                cancer_type,
                model_version,
                AVG(auc) as current_auc,
                AVG(sensitivity) as current_sensitivity
            FROM clinical_metrics
            WHERE metric_date >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY cancer_type, model_version
            """

            cursor.execute(query)
            current_metrics = cursor.fetchall()

            # Compare with baseline (30-day average)
            baseline_query = """
            SELECT
                cancer_type,
                model_version,
                AVG(auc) as baseline_auc,
                AVG(sensitivity) as baseline_sensitivity
            FROM clinical_metrics
            WHERE metric_date >= CURRENT_DATE - INTERVAL '37 days'
              AND metric_date < CURRENT_DATE - INTERVAL '7 days'
            GROUP BY cancer_type, model_version
            """

            cursor.execute(baseline_query)
            baseline_metrics = cursor.fetchall()

            # Calculate drift scores
            baseline_dict = {
                (row[0], row[1]): (row[2], row[3])
                for row in baseline_metrics
            }

            for cancer_type, model_version, current_auc, current_sensitivity in current_metrics:
                key = (cancer_type, model_version)
                if key in baseline_dict:
                    baseline_auc, baseline_sensitivity = baseline_dict[key]

                    # Calculate drift as relative change
                    auc_drift = abs(current_auc - baseline_auc) / baseline_auc if baseline_auc > 0 else 0
                    sensitivity_drift = abs(current_sensitivity - baseline_sensitivity) / baseline_sensitivity if baseline_sensitivity > 0 else 0

                    # Combined drift score
                    drift_score = (auc_drift + sensitivity_drift) / 2

                    self.model_drift_score.labels(
                        cancer_type=cancer_type,
                        model_version=model_version
                    ).set(drift_score)

        except Exception as e:
            logger.error(f"Failed to collect model drift metrics: {e}")

    async def collect_physician_agreement_metrics(self):
        """Collect physician agreement metrics."""
        if not self.db_conn:
            return

        try:
            cursor = self.db_conn.cursor()

            # Calculate physician agreement rates
            query = """
            SELECT
                cancer_type,
                COUNT(*) as total_reviewed,
                COUNT(CASE WHEN agreement_level = 'agree' THEN 1 END) as agreed
            FROM predictions
            WHERE physician_reviewed = true
              AND created_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY cancer_type
            """

            cursor.execute(query)
            results = cursor.fetchall()

            for cancer_type, total_reviewed, agreed in results:
                agreement_rate = agreed / total_reviewed if total_reviewed > 0 else 0

                self.physician_agreement_rate.labels(
                    cancer_type=cancer_type,
                    time_period='30_days'
                ).set(agreement_rate)

        except Exception as e:
            logger.error(f"Failed to collect physician agreement metrics: {e}")

    async def update_metrics(self):
        """Update all metrics."""
        logger.info("Updating Medical AI metrics...")

        await asyncio.gather(
            self.collect_clinical_metrics(),
            self.collect_business_metrics(),
            self.collect_system_health_metrics(),
            self.collect_model_drift_metrics(),
            self.collect_physician_agreement_metrics()
        )

        logger.info("Metrics update completed")

    async def run(self):
        """Run the metrics exporter."""
        # Connect to databases
        await self.connect_databases()

        # Start Prometheus HTTP server
        port = int(os.getenv('METRICS_PORT', '8001'))
        start_http_server(port)
        logger.info(f"Medical AI metrics exporter listening on port {port}")

        # Update metrics periodically
        update_interval = int(os.getenv('METRICS_UPDATE_INTERVAL', '60'))  # seconds

        while True:
            try:
                await self.update_metrics()
                await asyncio.sleep(update_interval)
            except Exception as e:
                logger.error(f"Metrics update failed: {e}")
                await asyncio.sleep(10)  # Retry sooner on failure

async def main():
    """Main function."""
    exporter = MedicalAIMetricsExporter()
    await exporter.run()

if __name__ == "__main__":
    asyncio.run(main())