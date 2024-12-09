import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from .paths import (
    DATASET_ROOT, PROJECT_ROOT, MODELS_PATH,
    RESULTS_PATH, LOGS_PATH, CHECKPOINTS_PATH,
    DRIVE_PATH
)

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    warmup_epochs: int = 2
    early_stopping_patience: int = 5
    validation_interval: int = 1
    mixed_precision: bool = True
    
    def validate(self):
        """Validate training configuration."""
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.weight_decay >= 0, "Weight decay must be non-negative"
        assert self.gradient_clip_val > 0, "Gradient clip value must be positive"
        assert self.warmup_epochs >= 0, "Warmup epochs must be non-negative"
        assert self.early_stopping_patience > 0, "Early stopping patience must be positive"
        assert self.validation_interval > 0, "Validation interval must be positive"

@dataclass
class ModelConfig:
    architecture: str = 'efficientnet'
    pretrained: bool = True
    num_classes: int = 2
    dropout_rate: float = 0.3
    label_smoothing: float = 0.1
    gradient_checkpointing: bool = False
    feature_visualization: bool = True
    
    def validate(self):
        """Validate model configuration."""
        valid_architectures = ['efficientnet', 'swin', 'two_stream', 'xception', 
                             'cnn_transformer', 'cross_attention']
        assert self.architecture in valid_architectures, f"Architecture must be one of {valid_architectures}"
        assert self.num_classes > 1, "Number of classes must be greater than 1"
        assert 0 <= self.dropout_rate <= 1, "Dropout rate must be between 0 and 1"
        assert 0 <= self.label_smoothing < 1, "Label smoothing must be between 0 and 1"

@dataclass
class DataConfig:
    image_size: int = 224
    num_workers: int = 4
    pin_memory: bool = True
    cache_size: int = 1000  # Number of images to cache in memory
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    def validate(self):
        """Validate data configuration."""
        assert self.image_size > 0, "Image size must be positive"
        assert self.num_workers >= 0, "Number of workers must be non-negative"
        assert self.cache_size >= 0, "Cache size must be non-negative"
        assert self.prefetch_factor > 0, "Prefetch factor must be positive"

@dataclass
class Config:
    version: str = "2.0.0"
    base_path: Path = field(default_factory=lambda: Path(PROJECT_ROOT))
    use_drive: bool = True
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Initialize and validate configuration."""
        # Convert string path to Path object
        self.base_path = Path(self.base_path)
        
        # Setup paths
        self.paths = {
            'data': Path(DATASET_ROOT),
            'models': Path(MODELS_PATH),
            'results': Path(RESULTS_PATH),
            'logs': Path(LOGS_PATH),
            'checkpoints': Path(CHECKPOINTS_PATH),
            'drive': Path(DRIVE_PATH) if self.use_drive else None
        }
        
        # Create directories
        for path in self.paths.values():
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)
                
        # Validate configurations
        self.validate()
        
    def validate(self):
        """Validate entire configuration."""
        self.training.validate()
        self.model.validate()
        self.data.validate()
        
        # Validate paths
        for name, path in self.paths.items():
            if path is not None and not path.exists():
                raise ValueError(f"{name} path does not exist: {path}")
                
    def save(self, path: Optional[str] = None):
        """Save configuration to JSON file."""
        if path is None:
            path = self.paths['results'] / 'config.json'
            
        # Convert paths to strings for JSON serialization
        config_dict = {
            'version': self.version,
            'base_path': str(self.base_path),
            'use_drive': self.use_drive,
            'training': self.training.__dict__,
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'created_at': self.created_at,
            'paths': {k: str(v) if v is not None else None for k, v in self.paths.items()}
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
            
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
            
        # Create configs from dictionaries
        training_config = TrainingConfig(**config_dict['training'])
        model_config = ModelConfig(**config_dict['model'])
        data_config = DataConfig(**config_dict['data'])
        
        # Create main config
        config = cls(
            version=config_dict['version'],
            base_path=config_dict['base_path'],
            use_drive=config_dict['use_drive'],
            training=training_config,
            model=model_config,
            data=data_config
        )
        
        return config
        
    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Try to update nested configs
                for config in [self.training, self.model, self.data]:
                    if hasattr(config, key):
                        setattr(config, key, value)
                        break
                else:
                    raise ValueError(f"Unknown configuration parameter: {key}")
        
        # Revalidate after update
        self.validate()