"""
Cancer-Specific Data Loaders

Provides data loading and preprocessing for different cancer types:
- Lung cancer (CT scans)
- Breast cancer (mammograms)
- Support for downloading and organizing datasets
"""

import os
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
import logging
import zipfile
import tarfile

logger = logging.getLogger(__name__)


class CancerDataLoader:
    """
    Base class for cancer-specific data loaders.

    Provides common functionality for:
    - Dataset downloading
    - Data organization
    - Patient-wise splitting
    - Metadata management
    """

    def __init__(self, cancer_type: str, config_path: str = "config.yaml"):
        """Initialize data loader."""
        self.cancer_type = cancer_type

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up paths
        self.raw_data_path = Path(self.config['paths']['raw_data']) / cancer_type
        self.processed_data_path = Path(self.config['paths']['processed_data']) / cancer_type
        self.splits_path = Path(self.config['paths']['splits'])

        # Create directories
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.splits_path.mkdir(parents=True, exist_ok=True)

        # Get cancer configuration
        self.cancer_config = None
        for cancer in self.config['cancer_types']:
            if cancer['name'] == cancer_type:
                self.cancer_config = cancer
                break

        if self.cancer_config is None:
            raise ValueError(f"Cancer type '{cancer_type}' not found in configuration")

    def download_data(self) -> bool:
        """Download dataset. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement download_data()")

    def process_data(self) -> pd.DataFrame:
        """Process raw data into standardized format. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement process_data()")

    def create_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create patient-wise train/validation/test splits."""
        if 'patient_id' not in df.columns:
            logger.warning("No patient_id column found. Creating random splits.")
            return self._create_random_splits(df)

        # Patient-wise splitting to prevent data leakage
        unique_patients = df['patient_id'].unique()
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(unique_patients)

        n_patients = len(unique_patients)
        train_end = int(n_patients * self.config['data']['train_split'])
        val_end = train_end + int(n_patients * self.config['data']['val_split'])

        train_patients = unique_patients[:train_end]
        val_patients = unique_patients[train_end:val_end]
        test_patients = unique_patients[val_end:]

        # Create split DataFrames
        splits = {}
        splits['train'] = df[df['patient_id'].isin(train_patients)].copy()
        splits['val'] = df[df['patient_id'].isin(val_patients)].copy()
        splits['test'] = df[df['patient_id'].isin(test_patients)].copy()

        logger.info(f"Created splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

        return splits

    def _create_random_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create random splits when patient IDs are not available."""
        n_samples = len(df)
        indices = np.random.permutation(n_samples)

        train_end = int(n_samples * self.config['data']['train_split'])
        val_end = train_end + int(n_samples * self.config['data']['val_split'])

        splits = {
            'train': df.iloc[indices[:train_end]].copy(),
            'val': df.iloc[indices[train_end:val_end]].copy(),
            'test': df.iloc[indices[val_end:]].copy()
        }

        return splits

    def save_splits(self, splits: Dict[str, pd.DataFrame]):
        """Save split DataFrames to CSV files."""
        for split_name, df in splits.items():
            output_path = self.splits_path / f"{self.cancer_type}_{split_name}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {split_name} split to {output_path}")

    def _download_file(self, url: str, filename: str, chunk_size: int = 8192) -> str:
        """Download file with progress bar."""
        filepath = self.raw_data_path / filename

        if filepath.exists():
            logger.info(f"File {filename} already exists. Skipping download.")
            return str(filepath)

        logger.info(f"Downloading {filename} from {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                pbar.update(size)

        return str(filepath)

    def _extract_archive(self, archive_path: str, extract_to: Optional[str] = None) -> str:
        """Extract zip or tar archives."""
        if extract_to is None:
            extract_to = str(Path(archive_path).parent)

        archive_path = Path(archive_path)

        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.bz2']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

        logger.info(f"Extracted {archive_path} to {extract_to}")
        return extract_to


class LungCancerLoader(CancerDataLoader):
    """
    Data loader for lung cancer detection from CT scans.

    Supports datasets like LIDC-IDRI for pulmonary nodule detection.
    """

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__('lung', config_path)

    def download_data(self) -> bool:
        """
        Download LIDC-IDRI dataset or similar lung CT datasets.

        Note: LIDC-IDRI requires registration and manual download.
        This method provides instructions and sample data structure.
        """
        logger.info("LIDC-IDRI dataset requires manual download from:")
        logger.info("https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI")
        logger.info("")
        logger.info("Please download and extract to: {}".format(self.raw_data_path))
        logger.info("Expected structure:")
        logger.info("lung/")
        logger.info("├── LIDC-IDRI/")
        logger.info("│   ├── LIDC-IDRI-0001/")
        logger.info("│   │   ├── 0001.dcm")
        logger.info("│   │   └── ...")
        logger.info("│   └── ...")

        # For demonstration, create sample data structure
        self._create_sample_lung_data()

        return True

    def _create_sample_lung_data(self):
        """Create sample lung cancer data for demonstration."""
        logger.info("Creating sample lung cancer data structure...")

        # Create sample annotations
        sample_data = []
        classes = self.cancer_config['classes']

        # Simulate 100 patients with varying numbers of scans
        np.random.seed(42)
        for patient_id in range(1, 101):
            n_scans = np.random.randint(1, 5)  # 1-4 scans per patient

            for scan_id in range(n_scans):
                # Simulate class distribution (more benign than malignant)
                label = np.random.choice(classes, p=[0.7, 0.3])

                sample_data.append({
                    'patient_id': f'LIDC-{patient_id:04d}',
                    'image_path': f'LIDC-IDRI/LIDC-IDRI-{patient_id:04d}/scan_{scan_id:03d}.png',
                    'label': label,
                    'modality': 'CT',
                    'dataset': 'LIDC-IDRI'
                })

        # Save sample annotations
        df = pd.DataFrame(sample_data)
        annotations_path = self.raw_data_path / 'lung_annotations.csv'
        df.to_csv(annotations_path, index=False)

        # Create dummy image files (small placeholders)
        for _, row in df.iterrows():
            img_path = self.raw_data_path / row['image_path']
            img_path.parent.mkdir(parents=True, exist_ok=True)

            # Create a dummy 224x224 RGB image
            dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(dummy_image).save(img_path)

        logger.info(f"Created sample dataset with {len(df)} images")

    def process_data(self) -> pd.DataFrame:
        """Process LIDC-IDRI data into standardized format."""
        annotations_path = self.raw_data_path / 'lung_annotations.csv'

        if not annotations_path.exists():
            logger.error("Annotations file not found. Run download_data() first.")
            return pd.DataFrame()

        df = pd.read_csv(annotations_path)

        # Standardize column names
        df = df.rename(columns={
            'image_path': 'image_path',
            'label': 'label',
            'patient_id': 'patient_id'
        })

        # Add relative path from processed data directory
        df['image_path'] = df['image_path'].apply(
            lambda x: str(Path('lung') / x)
        )

        # Validate data
        logger.info(f"Processed {len(df)} lung cancer images")
        logger.info(f"Class distribution: {df['label'].value_counts().to_dict()}")

        return df


class BreastCancerLoader(CancerDataLoader):
    """
    Data loader for breast cancer detection from mammograms.

    Supports datasets like CBIS-DDSM for mammogram analysis.
    """

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__('breast', config_path)

    def download_data(self) -> bool:
        """
        Download CBIS-DDSM dataset or similar breast imaging datasets.

        Note: CBIS-DDSM requires registration and manual download.
        This method provides instructions and sample data structure.
        """
        logger.info("CBIS-DDSM dataset requires manual download from:")
        logger.info("https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM")
        logger.info("")
        logger.info("Please download and extract to: {}".format(self.raw_data_path))
        logger.info("Expected structure:")
        logger.info("breast/")
        logger.info("├── CBIS-DDSM/")
        logger.info("│   ├── benign/")
        logger.info("│   │   ├── patient_001.png")
        logger.info("│   │   └── ...")
        logger.info("│   └── malignant/")
        logger.info("│       ├── patient_050.png")
        logger.info("│       └── ...")

        # For demonstration, create sample data structure
        self._create_sample_breast_data()

        return True

    def _create_sample_breast_data(self):
        """Create sample breast cancer data for demonstration."""
        logger.info("Creating sample breast cancer data structure...")

        # Create sample annotations
        sample_data = []
        classes = self.cancer_config['classes']

        # Simulate 200 patients
        np.random.seed(42)
        for patient_id in range(1, 201):
            # Simulate class distribution
            label = np.random.choice(classes, p=[0.6, 0.4])  # More benign cases

            sample_data.append({
                'patient_id': f'CBIS-{patient_id:04d}',
                'image_path': f'CBIS-DDSM/{label}/patient_{patient_id:04d}.png',
                'label': label,
                'modality': 'mammogram',
                'dataset': 'CBIS-DDSM',
                'birads_score': np.random.randint(1, 6) if label == 'malignant' else np.random.randint(1, 4)
            })

        # Save sample annotations
        df = pd.DataFrame(sample_data)
        annotations_path = self.raw_data_path / 'breast_annotations.csv'
        df.to_csv(annotations_path, index=False)

        # Create dummy image files
        for _, row in df.iterrows():
            img_path = self.raw_data_path / row['image_path']
            img_path.parent.mkdir(parents=True, exist_ok=True)

            # Create a dummy 224x224 grayscale image (mammograms are often grayscale)
            dummy_image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(dummy_image, mode='L').save(img_path)

        logger.info(f"Created sample dataset with {len(df)} images")

    def process_data(self) -> pd.DataFrame:
        """Process CBIS-DDSM data into standardized format."""
        annotations_path = self.raw_data_path / 'breast_annotations.csv'

        if not annotations_path.exists():
            logger.error("Annotations file not found. Run download_data() first.")
            return pd.DataFrame()

        df = pd.read_csv(annotations_path)

        # Standardize column names
        df = df.rename(columns={
            'image_path': 'image_path',
            'label': 'label',
            'patient_id': 'patient_id'
        })

        # Add relative path from processed data directory
        df['image_path'] = df['image_path'].apply(
            lambda x: str(Path('breast') / x)
        )

        # Validate data
        logger.info(f"Processed {len(df)} breast cancer images")
        logger.info(f"Class distribution: {df['label'].value_counts().to_dict()}")

        return df


def prepare_all_cancer_data(cancer_types: List[str] = None,
                           config_path: str = "config.yaml") -> Dict[str, pd.DataFrame]:
    """
    Prepare data for multiple cancer types.

    Args:
        cancer_types: List of cancer types to prepare. If None, uses all from config.
        config_path: Path to configuration file.

    Returns:
        Dictionary mapping cancer types to their processed DataFrames.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if cancer_types is None:
        cancer_types = [cancer['name'] for cancer in config['cancer_types']]

    # Initialize loaders
    loaders = {
        'lung': LungCancerLoader,
        'breast': BreastCancerLoader,
        # Add more loaders as they are implemented
    }

    processed_data = {}

    for cancer_type in cancer_types:
        if cancer_type not in loaders:
            logger.warning(f"No loader available for cancer type: {cancer_type}")
            continue

        logger.info(f"Preparing data for {cancer_type} cancer...")

        # Initialize loader
        loader_class = loaders[cancer_type]
        loader = loader_class(config_path)

        # Download data
        loader.download_data()

        # Process data
        df = loader.process_data()

        if len(df) > 0:
            # Create splits
            splits = loader.create_splits(df)

            # Save splits
            loader.save_splits(splits)

            processed_data[cancer_type] = df

        logger.info(f"Completed data preparation for {cancer_type}")

    return processed_data