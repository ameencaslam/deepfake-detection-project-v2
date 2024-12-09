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
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

@dataclass
class ModelConfig:
    architecture: str = 'efficientnet'
    pretrained: bool = True
    num_classes: int = 2
    dropout_rate: float = 0.3

@dataclass
class DataConfig:
    image_size: int = 224
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class Config:
    base_path: str = PROJECT_ROOT
    use_drive: bool = True
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def __post_init__(self):
        self.paths = {
            'data': DATASET_ROOT,
            'models': MODELS_PATH,
            'results': RESULTS_PATH,
            'logs': LOGS_PATH,
            'checkpoints': CHECKPOINTS_PATH,
            'drive': DRIVE_PATH
        }