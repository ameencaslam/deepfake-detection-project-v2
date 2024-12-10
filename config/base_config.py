import os
from dataclasses import dataclass, field
from typing import Dict, Any
from .paths import (
    DATASET_ROOT, PROJECT_ROOT, MODELS_PATH,
    RESULTS_PATH, LOGS_PATH, CHECKPOINTS_PATH,
    DRIVE_PATH
)

@dataclass
class TrainingConfig:
    """Training configuration."""
    def __init__(self):
        self.batch_size = 32
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.scheduler = 'onecycle'  # onecycle, cosine, linear
        self.warmup_epochs = 5
        self.mixed_precision = True

@dataclass
class ModelConfig:
    """Model configuration."""
    def __init__(self):
        # Basic settings
        self.architecture = 'xception'  # xception, cnn_transformer, cross_attention, two_stream
        self.pretrained = True
        self.num_classes = 2
        self.dropout_rate = 0.3
        self.label_smoothing = 0.1
        
        # Architecture-specific settings
        self.hidden_dim = 512
        self.num_heads = 8
        self.num_layers = 3
        self.use_attention = True

@dataclass
class DataConfig:
    """Data configuration."""
    def __init__(self):
        self.image_size = 224
        self.num_workers = 4
        self.pin_memory = True
        self.prefetch_factor = 2

@dataclass
class Config:
    """Main configuration class."""
    def __init__(self, base_path: str = None, use_drive: bool = False):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        
        # Set up paths
        if base_path is None:
            base_path = os.getcwd()
        
        self.paths = {
            'base': base_path,
            'data': os.path.join(base_path, 'data'),
            'checkpoints': os.path.join(base_path, 'checkpoints'),
            'results': os.path.join(base_path, 'results'),
            'logs': os.path.join(base_path, 'logs')
        }
        
        # Create directories
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)