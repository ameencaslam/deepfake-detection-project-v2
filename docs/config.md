# Configuration System Documentation

## Overview

The configuration system manages all settings for training, data processing, and model architectures.

## Files

- `config/base_config.py`: Main configuration class
- `config/model_configs/`: Model-specific configurations

## Configuration Classes

### TrainingConfig

```python
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
```

### DataConfig

```python
@dataclass
class DataConfig:
    image_size: int = 224
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4
    pin_memory: bool = True
```

### ModelConfig

```python
@dataclass
class ModelConfig:
    architecture: str = "efficientnet_b3"
    pretrained: bool = True
    dropout_rate: float = 0.2
    num_classes: int = 2
```

## Usage

1. Basic Usage:

```python
from config.base_config import Config

config = Config(base_path='/path/to/project')
```

2. Updating Configuration:

```python
config.update(
    batch_size=64,
    learning_rate=1e-3
)
```

3. Path Management:

```python
# Paths are automatically created
print(config.paths['data'])      # Data directory
print(config.paths['models'])    # Model checkpoints
print(config.paths['results'])   # Training results
```

## Drive Integration

- Set `use_drive=True` for Google Drive integration
- Drive paths are automatically managed
- Checkpoints and results are backed up to Drive

## Hardware Configuration

- Automatic device detection (CPU/GPU/TPU)
- Mixed precision settings
- Memory optimization

## Important Notes

1. All paths are created automatically
2. Drive integration requires Google Colab environment
3. Configuration is saved with checkpoints
4. Parameters can be modified during runtime
