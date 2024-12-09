# Quickstart Guide

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training Models

### Using main.py (Recommended)

```bash
# Train with default settings (example using EfficientNet)
python main.py --model efficientnet --drive True

# Train with custom batch size
python main.py --model MODEL_NAME --drive True --batch 64
```

### Using train.py (Advanced)

```python
from config.base_config import Config
from train import train

# Configure and train any supported model
config = Config(base_path='project_path', use_drive=True)
config.model.architecture = 'MODEL_NAME'  # Set desired model
train(config)
```

## Model Evaluation

```bash
# Evaluate using latest checkpoint
python evaluate.py --model MODEL_NAME

# Custom evaluation
python evaluate.py \
    --model MODEL_NAME \
    --checkpoint path/to/checkpoint.pth \
    --data_path path/to/test/data \
    --batch_size 32
```

## Project Management

The `manage.py` script handles project state:

```bash
# Backup project state
python manage.py backup

# Restore from backup
python manage.py restore

# Clean temporary files
python manage.py clean
```

## Directory Structure

```
project/
├── config/             # Configuration
├── models/            # Model architectures
├── utils/             # Utilities
├── docs/              # Documentation
├── main.py           # Main script
├── train.py          # Training script
├── evaluate.py       # Evaluation script
├── manage.py         # Project management
└── requirements.txt  # Dependencies
```

## Configuration

Edit `config/base_config.py` for:

- Model parameters
- Training settings
- Dataset options
- Hardware settings

## Common Operations

### Training

```bash
# Basic training
python main.py --model MODEL_NAME --drive True

# Resume from checkpoint
python main.py --model MODEL_NAME --drive True --resume
```

### Evaluation

```bash
# Basic evaluation
python evaluate.py --model MODEL_NAME

# Detailed evaluation
python evaluate.py --model MODEL_NAME --data_path test_data
```

### Project Management

```bash
# After training session
python manage.py backup

# Before new session
python manage.py restore

# Clean temporary files
python manage.py clean
```
