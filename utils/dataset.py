import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import numpy as np
from pathlib import Path
import json
import cv2
import logging
from dataclasses import dataclass, field
import random
from PIL import Image
import h5py
from tqdm.auto import tqdm
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import partial
import warnings

@dataclass
class DatasetMetadata:
    """Dataset metadata and statistics."""
    name: str
    version: str
    num_samples: int
    num_classes: int
    class_distribution: Dict[str, int]
    image_size: Tuple[int, int]
    channels: int
    mean: np.ndarray
    std: np.ndarray
    split_info: Dict[str, List[str]]
    augmentations: List[str]
    cache_info: Dict[str, Any] = field(default_factory=dict)

class DatasetCache:
    """Efficient dataset caching system."""
    
    def __init__(self, cache_dir: Union[str, Path], capacity_gb: float = 10.0):
        """Initialize cache.
        
        Args:
            cache_dir: Cache directory
            capacity_gb: Cache capacity in gigabytes
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.capacity_bytes = int(capacity_gb * 1e9)
        self.current_size = 0
        self.cache = {}
        self.access_count = {}
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                self.access_count[key] += 1
                return self.cache[key]
        return None
        
    def put(self, key: str, value: np.ndarray):
        """Put item in cache."""
        item_size = value.nbytes
        
        with self.lock:
            # Check if item already exists
            if key in self.cache:
                self.cache[key] = value
                return
                
            # Remove least accessed items if needed
            while self.current_size + item_size > self.capacity_bytes and self.cache:
                least_key = min(self.access_count.items(), key=lambda x: x[1])[0]
                self._remove_item(least_key)
                
            # Add new item
            self.cache[key] = value
            self.access_count[key] = 1
            self.current_size += item_size
            
    def _remove_item(self, key: str):
        """Remove item from cache."""
        item_size = self.cache[key].nbytes
        del self.cache[key]
        del self.access_count[key]
        self.current_size -= item_size
        
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()
            self.current_size = 0

class AugmentationPipeline:
    """Advanced data augmentation pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize augmentation pipeline.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()
        
    def _create_train_transform(self) -> A.Compose:
        """Create training augmentation pipeline."""
        transforms_list = [
            A.RandomResizedCrop(
                height=self.config['image_size'],
                width=self.config['image_size'],
                scale=(0.8, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightness(limit=0.2),
                A.RandomContrast(limit=0.2),
                A.RandomGamma(gamma_limit=(80, 120))
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(color_shift=(0.01, 0.05)),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1))
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5)
            ], p=0.2),
            A.OneOf([
                A.JpegCompression(quality_lower=80),
                A.Downscale(scale_min=0.8, scale_max=0.9),
                A.ImageCompression(quality_lower=80)
            ], p=0.2),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.3
            ),
            A.Normalize(
                mean=self.config.get('mean', [0.485, 0.456, 0.406]),
                std=self.config.get('std', [0.229, 0.224, 0.225])
            ),
            ToTensorV2()
        ]
        
        if self.config.get('cutout', False):
            transforms_list.insert(-2, A.CoarseDropout(
                max_holes=8,
                max_height=8,
                max_width=8,
                fill_value=0,
                p=0.2
            ))
            
        if self.config.get('mixup', False):
            self.mixup = True
            self.mixup_alpha = self.config.get('mixup_alpha', 0.2)
        else:
            self.mixup = False
            
        return A.Compose(transforms_list)
        
    def _create_val_transform(self) -> A.Compose:
        """Create validation augmentation pipeline."""
        return A.Compose([
            A.Resize(
                height=self.config['image_size'],
                width=self.config['image_size']
            ),
            A.Normalize(
                mean=self.config.get('mean', [0.485, 0.456, 0.406]),
                std=self.config.get('std', [0.229, 0.224, 0.225])
            ),
            ToTensorV2()
        ])
        
    def __call__(self, image: np.ndarray, train: bool = True) -> torch.Tensor:
        """Apply augmentation pipeline."""
        if train:
            return self.train_transform(image=image)['image']
        return self.val_transform(image=image)['image']
        
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation."""
        if not self.mixup:
            return x, y, 1.0
            
        batch_size = x.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y, lam

class DeepfakeDataset(Dataset):
    """Enhanced deepfake detection dataset."""
    
    def __init__(self,
                 data_dir: Union[str, Path],
                 split: str = 'train',
                 transform_config: Optional[Dict[str, Any]] = None,
                 cache_dir: Optional[Union[str, Path]] = None,
                 cache_capacity_gb: float = 10.0,
                 num_workers: int = 4):
        """Initialize dataset.
        
        Args:
            data_dir: Data directory
            split: Dataset split ('train', 'val', 'test')
            transform_config: Transform configuration
            cache_dir: Cache directory
            cache_capacity_gb: Cache capacity in gigabytes
            num_workers: Number of workers for parallel processing
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform_config = transform_config or {'image_size': 224}
        self.num_workers = num_workers
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Load or create split info
        self.split_file = self.data_dir / 'split_info.json'
        self.split_info = self._load_split_info()
        
        # Setup transforms
        self.transform = AugmentationPipeline(self.transform_config)
        
        # Setup cache
        if cache_dir is not None:
            self.cache = DatasetCache(cache_dir, cache_capacity_gb)
        else:
            self.cache = None
            
        # Load dataset
        self.samples = self.split_info[split]
        self.labels = [1 if 'fake' in str(p) else 0 for p in self.samples]
        
        # Calculate statistics
        self.metadata = self._calculate_metadata()
        
        # Preload data if cache is enabled
        if self.cache is not None:
            self._preload_data()
            
    def _load_split_info(self) -> Dict[str, List[str]]:
        """Load or create split information."""
        if self.split_file.exists():
            with open(self.split_file, 'r') as f:
                return json.load(f)
                
        # Create new split info
        all_files = []
        for label in ['real', 'fake']:
            files = list((self.data_dir / label).rglob('*.jpg'))
            all_files.extend([str(f.relative_to(self.data_dir)) for f in files])
            
        # Shuffle files
        random.shuffle(all_files)
        
        # Create splits
        total = len(all_files)
        train_idx = int(0.8 * total)
        val_idx = int(0.9 * total)
        
        split_info = {
            'train': all_files[:train_idx],
            'val': all_files[train_idx:val_idx],
            'test': all_files[val_idx:]
        }
        
        # Save split info
        with open(self.split_file, 'w') as f:
            json.dump(split_info, f, indent=4)
            
        return split_info
        
    def _calculate_metadata(self) -> DatasetMetadata:
        """Calculate dataset metadata and statistics."""
        # Count classes
        class_counts = {'real': 0, 'fake': 0}
        for label in self.labels:
            class_counts['fake' if label == 1 else 'real'] += 1
            
        # Calculate mean and std if not provided
        if 'mean' not in self.transform_config or 'std' not in self.transform_config:
            mean, std = self._calculate_normalization_stats()
            self.transform_config['mean'] = mean.tolist()
            self.transform_config['std'] = std.tolist()
            
        return DatasetMetadata(
            name='DeepfakeDataset',
            version='2.0.0',
            num_samples=len(self.samples),
            num_classes=2,
            class_distribution=class_counts,
            image_size=(self.transform_config['image_size'],
                       self.transform_config['image_size']),
            channels=3,
            mean=np.array(self.transform_config['mean']),
            std=np.array(self.transform_config['std']),
            split_info={k: len(v) for k, v in self.split_info.items()},
            augmentations=list(self.transform_config.keys())
        )
        
    def _calculate_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate dataset mean and standard deviation."""
        self.logger.info("Calculating dataset statistics...")
        
        # Use subset of images for speed
        subset_size = min(1000, len(self.samples))
        subset_indices = random.sample(range(len(self.samples)), subset_size)
        
        # Calculate in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(
                executor.map(self._load_image, [self.samples[i] for i in subset_indices]),
                total=subset_size,
                desc="Calculating statistics"
            ))
            
        # Stack results
        images = np.stack([r for r in results if r is not None])
        
        # Calculate mean and std
        mean = np.mean(images, axis=(0, 1, 2))
        std = np.std(images, axis=(0, 1, 2))
        
        return mean, std
        
    def _preload_data(self):
        """Preload data into cache."""
        self.logger.info("Preloading dataset into cache...")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            list(tqdm(
                executor.map(self._cache_sample, range(len(self))),
                total=len(self),
                desc="Preloading data"
            ))
            
    def _cache_sample(self, idx: int):
        """Cache a single sample."""
        image_path = self.samples[idx]
        cache_key = self._get_cache_key(image_path)
        
        if self.cache.get(cache_key) is None:
            image = self._load_image(image_path)
            if image is not None:
                self.cache.put(cache_key, image)
                
    def _get_cache_key(self, path: str) -> str:
        """Generate cache key for path."""
        return hashlib.md5(path.encode()).hexdigest()
        
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from path."""
        try:
            image = cv2.imread(str(self.data_dir / image_path))
            if image is None:
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.logger.warning(f"Error loading image {image_path}: {e}")
            return None
            
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get dataset item."""
        image_path = self.samples[idx]
        label = self.labels[idx]
        
        # Try to get from cache first
        if self.cache is not None:
            cache_key = self._get_cache_key(image_path)
            image = self.cache.get(cache_key)
            
            if image is None:
                image = self._load_image(image_path)
                if image is not None:
                    self.cache.put(cache_key, image)
        else:
            image = self._load_image(image_path)
            
        if image is None:
            # Return a zero tensor with correct shape if image loading fails
            return torch.zeros((3, self.transform_config['image_size'],
                              self.transform_config['image_size'])), label
                              
        # Apply transforms
        image = self.transform(image, train=(self.split == 'train'))
        
        return image, label
        
    def get_sampler(self, distributed: bool = False) -> Optional[torch.utils.data.Sampler]:
        """Get data sampler."""
        if distributed:
            return torch.utils.data.distributed.DistributedSampler(self)
            
        if self.split == 'train':
            # Use weighted sampler for class balancing
            class_weights = [
                1.0 / self.metadata.class_distribution['real'],
                1.0 / self.metadata.class_distribution['fake']
            ]
            weights = [class_weights[label] for label in self.labels]
            return torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(self),
                replacement=True
            )
            
        return None
        
    def get_dataloader(self,
                      batch_size: int,
                      shuffle: bool = None,
                      num_workers: int = None,
                      distributed: bool = False) -> DataLoader:
        """Get data loader."""
        if shuffle is None:
            shuffle = (self.split == 'train' and not distributed)
            
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers or self.num_workers,
            sampler=self.get_sampler(distributed),
            pin_memory=True,
            drop_last=(self.split == 'train')
        )
        
    def save_metadata(self, path: Optional[Union[str, Path]] = None):
        """Save dataset metadata."""
        if path is None:
            path = self.data_dir / 'dataset_metadata.json'
        else:
            path = Path(path)
            
        with open(path, 'w') as f:
            json.dump(self.metadata.__dict__, f, indent=4)
            
    def cleanup(self):
        """Cleanup dataset resources."""
        if self.cache is not None:
            self.cache.clear()
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

# Update exports at the end of the file
__all__ = ['DeepfakeDataset', 'get_dataloader', 'create_dataloaders'] 