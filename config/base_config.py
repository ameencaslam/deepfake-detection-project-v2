from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch
import os

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_val: float = 1.0
    validation_interval: int = 1
    early_stopping_patience: int = 5
    mixed_precision: bool = True

@dataclass
class DataConfig:
    image_size: int = 224
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4
    pin_memory: bool = True
    dataset_root: str = '/content/3body-filtered-v2-10k'
    real_path: str = None
    fake_path: str = None

@dataclass
class ModelConfig:
    architecture: str = "efficientnet_b3"
    pretrained: bool = True
    dropout_rate: float = 0.2
    num_classes: int = 2

class Config:
    def __init__(self, base_path: str, use_drive: bool = True):
        self.base_path = base_path
        self.use_drive = use_drive
        self.device = self._setup_device()
        
        # Training settings
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        
        # Setup data paths
        self._setup_data_paths()
        
        # Paths
        self.paths = self._setup_paths()
        
    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device("xpu")
        else:
            return torch.device("cpu")
            
    def _setup_data_paths(self):
        """Setup dataset paths."""
        if self.data.real_path is None:
            self.data.real_path = os.path.join(self.data.dataset_root, 'real')
        if self.data.fake_path is None:
            self.data.fake_path = os.path.join(self.data.dataset_root, 'fake')
            
    def _setup_paths(self) -> Dict[str, str]:
        # Drive paths (for essential files)
        drive_path = '/content/drive/MyDrive/deepfake_project' if self.use_drive else None
        
        paths = {
            'base': self.base_path,
            'drive': drive_path,
            'data': self.data.dataset_root,
            'models': os.path.join(self.base_path, 'models'),
            'results': os.path.join(self.base_path, 'results'),
            'logs': os.path.join(self.base_path, 'logs'),
        }
        
        # Create directories
        for path in paths.values():
            if path is not None:
                os.makedirs(path, exist_ok=True)
                
        return paths
        
    def update(self, **kwargs):
        """Update config parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid config parameter: {key}")
                
    def validate_paths(self) -> bool:
        """Validate that all required paths exist."""
        required_paths = [
            self.data.dataset_root,
            self.data.real_path,
            self.data.fake_path
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                raise ValueError(f"Required path does not exist: {path}")
                
        return True