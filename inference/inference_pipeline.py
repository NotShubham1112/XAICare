"""
Clinical Inference Pipeline for Multi-Cancer Detection

Provides advanced inference capabilities for clinical workflows:
- Multi-cancer batch processing
- Quality control and validation
- Integration with clinical systems
- Performance monitoring and logging
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import yaml
import logging
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import project modules
from .predictor import CancerPredictor

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Production-ready inference pipeline for clinical cancer detection.

    Supports multi-cancer batch processing with quality control,
    performance monitoring, and clinical workflow integration.
    """

    def __init__(self,
                 model_configs: Dict[str, str],
                 config_path: str = "config.yaml",
                 max_workers: int = 4,
                 enable_monitoring: bool = True):
        """
        Initialize inference pipeline.

        Args:
            model_configs: Dictionary mapping cancer types to model paths
            config_path: Path to configuration file
            max_workers: Maximum number of worker threads
            enable_monitoring: Whether to enable performance monitoring
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_configs = model_configs
        self.max_workers = max_workers
        self.enable_monitoring = enable_monitoring

        # Initialize predictors for each cancer type
        self.predictors = {}
        self._load_predictors()

        # Quality control settings
        self.quality_thresholds = {
            'min_confidence': 0.3,
            'max_processing_time': 30.0,  # seconds
            'min_image_quality': 0.5
        }

        # Performance monitoring
        self.performance_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_processing_time': 0.0,
            'cancer_type_distribution': {},
            'error_types': {}
        }

        logger.info(f"InferencePipeline initialized with {len(self.predictors)} cancer types")

    def _load_predictors(self):
        """Load predictors for each cancer type."""
        for cancer_type, model_path in self.model_configs.items():
            try:
                predictor = CancerPredictor(
                    model_path=model_path,
                    cancer_types=[cancer_type],
                    config_path=self.config_path
                )
                self.predictors[cancer_type] = predictor
                logger.info(f"Loaded predictor for {cancer_type}")
            except Exception as e:
                logger.error(f"Failed to load predictor for {cancer_type}: {e}")

    def process_clinical_batch(self,
                             image_batch: List[Dict[str, Any]],
                             priority_processing: bool = False) -> Dict[str, Any]:
        """
        Process a batch of clinical images with quality control.

        Args:
            image_batch: List of image data dictionaries
            priority_processing: Whether to use priority processing

        Returns:
            Batch processing results with quality metrics
        """
        start_time = time.time()
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Processing clinical batch {batch_id} with {len(image_batch)} images")

        # Validate batch
        validation_results = self._validate_batch(image_batch)
        if not validation_results['valid']:
            return {
                'batch_id': batch_id,
                'status': 'rejected',
                'reason': validation_results['reason'],
                'processing_time': time.time() - start_time
            }

        # Process batch
        if priority_processing:
            results = self._process_batch_priority(image_batch)
        else:
            results = self._process_batch_parallel(image_batch)

        # Quality control and post-processing
        processed_results = self._post_process_results(results, batch_id)

        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processed_results, processing_time)

        final_results = {
            'batch_id': batch_id,
            'status': 'completed',
            'processing_time': processing_time,
            'results': processed_results,
            'quality_metrics': self._calculate_batch_quality_metrics(processed_results),
            'performance_stats': self.performance_stats.copy()
        }

        logger.info(f"Batch {batch_id} completed in {processing_time:.2f}s")
        return final_results

    def _validate_batch(self, image_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate batch before processing."""
        if not image_batch:
            return {'valid': False, 'reason': 'Empty batch'}

        # Check required fields
        required_fields = ['image_path', 'cancer_type', 'patient_id']
        for item in image_batch:
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                return {
                    'valid': False,
                    'reason': f"Missing required fields: {missing_fields} in item {item.get('image_path', 'unknown')}"
                }

        # Check cancer type support
        unsupported_types = []
        for item in image_batch:
            cancer_type = item['cancer_type']
            if cancer_type not in self.predictors:
                unsupported_types.append(cancer_type)

        if unsupported_types:
            return {
                'valid': False,
                'reason': f"Unsupported cancer types: {list(set(unsupported_types))}"
            }

        # Check file existence
        missing_files = []
        for item in image_batch:
            if not Path(item['image_path']).exists():
                missing_files.append(item['image_path'])

        if missing_files:
            return {
                'valid': False,
                'reason': f"Missing image files: {missing_files[:5]}"  # Show first 5
            }

        return {'valid': True}

    def _process_batch_parallel(self, image_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch in parallel using thread pools."""
        results = []

        # Group by cancer type for efficient processing
        cancer_groups = {}
        for item in image_batch:
            cancer_type = item['cancer_type']
            if cancer_type not in cancer_groups:
                cancer_groups[cancer_type] = []
            cancer_groups[cancer_type].append(item)

        # Process each cancer type
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(cancer_groups))) as executor:
            future_to_cancer = {
                executor.submit(self._process_cancer_group, cancer_type, items): cancer_type
                for cancer_type, items in cancer_groups.items()
            }

            for future in as_completed(future_to_cancer):
                cancer_type = future_to_cancer[future]
                try:
                    cancer_results = future.result()
                    results.extend(cancer_results)
                except Exception as e:
                    logger.error(f"Failed to process {cancer_type}: {e}")
                    # Add error results for all items in this group
                    for item in cancer_groups[cancer_type]:
                        results.append({
                            'patient_id': item.get('patient_id'),
                            'cancer_type': cancer_type,
                            'status': 'error',
                            'error': str(e)
                        })

        return results

    def _process_cancer_group(self, cancer_type: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all images for a specific cancer type."""
        predictor = self.predictors[cancer_type]
        results = []

        for item in items:
            try:
                start_time = time.time()

                # Make prediction
                prediction = predictor.predict_single_image(
                    item['image_path'],
                    cancer_type,
                    generate_explanation=True,
                    save_visualizations=False
                )

                processing_time = time.time() - start_time

                # Add metadata
                prediction.update({
                    'patient_id': item.get('patient_id'),
                    'batch_item_id': item.get('id'),
                    'processing_time': processing_time,
                    'status': 'success'
                })

                results.append(prediction)

            except Exception as e:
                logger.error(f"Prediction failed for {item.get('image_path')}: {e}")
                results.append({
                    'patient_id': item.get('patient_id'),
                    'cancer_type': cancer_type,
                    'status': 'error',
                    'error': str(e),
                    'image_path': item.get('image_path')
                })

        return results

    def _process_batch_priority(self, image_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch with priority for urgent cases."""
        # Sort by priority (could be based on clinical urgency, patient risk factors, etc.)
        # For now, process in order but could be enhanced
        return self._process_batch_parallel(image_batch)

    def _post_process_results(self, results: List[Dict[str, Any]], batch_id: str) -> List[Dict[str, Any]]:
        """Post-process results with quality control and clinical validation."""
        processed_results = []

        for result in results:
            if result.get('status') == 'error':
                processed_results.append(result)
                continue

            # Quality control checks
            quality_issues = self._check_prediction_quality(result)

            if quality_issues:
                result['quality_flags'] = quality_issues
                result['requires_review'] = True
            else:
                result['quality_flags'] = []
                result['requires_review'] = False

            # Add clinical workflow integration
            result['clinical_workflow'] = self._get_clinical_workflow(result)

            # Add batch metadata
            result['batch_id'] = batch_id
            result['processed_at'] = datetime.now().isoformat()

            processed_results.append(result)

        return processed_results

    def _check_prediction_quality(self, result: Dict[str, Any]) -> List[str]:
        """Check prediction quality against thresholds."""
        issues = []

        confidence = result.get('confidence', 0)
        processing_time = result.get('processing_time', 0)

        if confidence < self.quality_thresholds['min_confidence']:
            issues.append(f"Low confidence ({confidence:.2f} < {self.quality_thresholds['min_confidence']})")

        if processing_time > self.quality_thresholds['max_processing_time']:
            issues.append(f"Slow processing ({processing_time:.1f}s > {self.quality_thresholds['max_processing_time']}s)")

        # Check for unusual predictions
        if result.get('risk_level') == 'HIGH' and confidence < 0.5:
            issues.append("High risk prediction with low confidence - requires review")

        return issues

    def _get_clinical_workflow(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get clinical workflow recommendations based on prediction."""
        risk_level = result.get('risk_level', 'LOW')
        cancer_type = result.get('cancer_type', 'unknown')
        confidence = result.get('confidence', 0)

        workflow = {
            'recommended_action': 'routine_followup',
            'priority': 'normal',
            'timeframe': 'standard',
            'specialist_consultation': False,
            'additional_imaging': False,
            'biopsy_recommended': False
        }

        if risk_level == 'HIGH':
            workflow.update({
                'recommended_action': 'urgent_consultation',
                'priority': 'high',
                'timeframe': 'within_1_week',
                'specialist_consultation': True,
                'additional_imaging': True
            })

            if confidence > 0.8:
                workflow['biopsy_recommended'] = True

        elif risk_level == 'MEDIUM':
            workflow.update({
                'recommended_action': 'expedited_evaluation',
                'priority': 'medium',
                'timeframe': 'within_2_weeks',
                'additional_imaging': True
            })

        # Cancer-specific adjustments
        if cancer_type == 'lung':
            workflow['primary_specialty'] = 'thoracic_surgery'
            workflow['secondary_specialty'] = 'pulmonology'
        elif cancer_type == 'breast':
            workflow['primary_specialty'] = 'breast_surgery'
            workflow['secondary_specialty'] = 'oncology'
        else:
            workflow['primary_specialty'] = 'oncology'

        return workflow

    def _calculate_batch_quality_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality metrics for the batch."""
        total_predictions = len(results)
        successful_predictions = sum(1 for r in results if r.get('status') == 'success')
        quality_issues = sum(1 for r in results if r.get('requires_review', False))

        if successful_predictions > 0:
            confidences = [r.get('confidence', 0) for r in results if r.get('status') == 'success']
            avg_confidence = np.mean(confidences)
            high_risk_predictions = sum(1 for r in results
                                      if r.get('risk_level') == 'HIGH' and r.get('status') == 'success')
        else:
            avg_confidence = 0.0
            high_risk_predictions = 0

        return {
            'total_predictions': total_predictions,
            'successful_predictions': successful_predictions,
            'success_rate': successful_predictions / total_predictions * 100,
            'quality_issues': quality_issues,
            'quality_issue_rate': quality_issues / total_predictions * 100,
            'average_confidence': avg_confidence,
            'high_risk_predictions': high_risk_predictions,
            'batch_quality_score': self._calculate_batch_quality_score(
                successful_predictions, quality_issues, avg_confidence, total_predictions
            )
        }

    def _calculate_batch_quality_score(self, successful: int, issues: int,
                                     avg_confidence: float, total: int) -> float:
        """Calculate overall batch quality score."""
        if total == 0:
            return 0.0

        success_rate = successful / total
        issue_rate = issues / total

        # Weighted quality score
        quality_score = (
            0.4 * success_rate +           # Success rate
            0.3 * (1 - issue_rate) +       # Quality issue rate (inverted)
            0.3 * min(avg_confidence, 1.0) # Confidence (capped at 1.0)
        )

        return quality_score

    def _update_performance_stats(self, results: List[Dict[str, Any]], processing_time: float):
        """Update performance statistics."""
        self.performance_stats['total_predictions'] += len(results)

        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = len(results) - successful

        self.performance_stats['successful_predictions'] += successful
        self.performance_stats['failed_predictions'] += failed

        # Update average processing time
        current_avg = self.performance_stats['average_processing_time']
        total_processed = (self.performance_stats['successful_predictions'] +
                          self.performance_stats['failed_predictions'])
        self.performance_stats['average_processing_time'] = (
            (current_avg * (total_processed - len(results)) + processing_time) / total_processed
        )

        # Update cancer type distribution
        for result in results:
            cancer_type = result.get('cancer_type', 'unknown')
            if cancer_type not in self.performance_stats['cancer_type_distribution']:
                self.performance_stats['cancer_type_distribution'][cancer_type] = 0
            self.performance_stats['cancer_type_distribution'][cancer_type] += 1

        # Update error types
        for result in results:
            if result.get('status') == 'error':
                error_type = result.get('error', 'unknown_error')
                if error_type not in self.performance_stats['error_types']:
                    self.performance_stats['error_types'][error_type] = 0
                self.performance_stats['error_types'][error_type] += 1

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        stats = self.performance_stats

        total_processed = stats['successful_predictions'] + stats['failed_predictions']
        success_rate = (stats['successful_predictions'] / total_processed * 100) if total_processed > 0 else 0

        report = {
            'summary': {
                'total_predictions': stats['total_predictions'],
                'success_rate': success_rate,
                'average_processing_time': stats['average_processing_time'],
                'error_rate': (stats['failed_predictions'] / total_processed * 100) if total_processed > 0 else 0
            },
            'cancer_type_distribution': stats['cancer_type_distribution'],
            'error_analysis': stats['error_types'],
            'system_health': 'good' if success_rate > 95 else 'warning' if success_rate > 90 else 'critical'
        }

        return report

    def export_batch_results(self,
                           results: Dict[str, Any],
                           output_path: str,
                           format: str = 'json') -> str:
        """
        Export batch results to file.

        Args:
            results: Batch processing results
            output_path: Output file path
            format: Export format ('json', 'csv')

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)

        if format == 'json':
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=convert_numpy)

        elif format == 'csv':
            # Flatten results for CSV export
            flattened_results = []

            for result in results.get('results', []):
                flat_result = {
                    'batch_id': results.get('batch_id'),
                    'patient_id': result.get('patient_id'),
                    'cancer_type': result.get('cancer_type'),
                    'prediction': result.get('prediction'),
                    'confidence': result.get('confidence'),
                    'risk_level': result.get('risk_level'),
                    'status': result.get('status'),
                    'processing_time': result.get('processing_time'),
                    'requires_review': result.get('requires_review', False)
                }
                flattened_results.append(flat_result)

            df = pd.DataFrame(flattened_results)
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Batch results exported to {output_path}")
        return str(output_path)

    def validate_system_health(self) -> Dict[str, Any]:
        """Validate system health and performance."""
        health_check = {
            'predictors_loaded': len(self.predictors),
            'expected_predictors': len(self.model_configs),
            'performance_stats': self.get_performance_report(),
            'quality_thresholds': self.quality_thresholds,
            'timestamp': datetime.now().isoformat()
        }

        # Check system health
        success_rate = health_check['performance_stats']['summary']['success_rate']
        if success_rate > 95:
            health_check['system_status'] = 'healthy'
        elif success_rate > 90:
            health_check['system_status'] = 'degraded'
        else:
            health_check['system_status'] = 'unhealthy'

        return health_check