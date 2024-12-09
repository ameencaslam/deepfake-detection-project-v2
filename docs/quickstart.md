# Quickstart Guide

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Models

### Using main.py (High-Level Interface)

```bash
# Train all models sequentially
python main.py --model all

# Train single model with Drive backup
python main.py --model efficientnet --drive True

# Train with custom batch size
python main.py --model swin --batch 64
```

### Using train.py (Detailed Control)

```bash
# Basic training
python train.py --model efficientnet --data_path /path/to/data

# Advanced training options
python train.py \
    --model efficientnet \
    --data_path /path/to/data \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 50
```

## Available Models

- Swin Transformer (`swin`)
- Two-Stream Network (`two_stream`)
- Xception (`xception`)
- CNN-Transformer Hybrid (`cnn_transformer`)
- Cross-Attention Model (`cross_attention`)
- EfficientNet-B3 (`efficientnet`)

## Evaluation

Simple evaluation (uses latest checkpoint):

```bash
python evaluate.py --model efficientnet
```

Custom evaluation:

```bash
python evaluate.py \
    --model efficientnet \
    --checkpoint path/to/checkpoint.pth \
    --data_path path/to/test/data \
    --batch_size 32
```

## Getting Help

View available options for any script:

```bash
# High-level training options
python main.py --help

# Detailed training options
python train.py --help

# Evaluation options
python evaluate.py --help
```

## Project Structure

```
project/
├── config/             # Configuration files
├── models/             # Model architectures
│   └── architectures/  # Individual model implementations
├── utils/             # Utility functions
├── docs/              # Documentation
├── main.py           # High-level training script
├── train.py          # Detailed training script
├── evaluate.py       # Evaluation script
└── requirements.txt  # Project dependencies
```

## Script Purposes

### main.py (High-Level Control)

- Train multiple models sequentially
- Google Drive integration for backups
- Simple command-line interface
- Project-wide settings
- Best for quick experiments

### train.py (Detailed Control)

- Single model training
- Fine-grained parameter control
- Advanced training features
- Custom training loops
- Best for detailed experimentation

### evaluate.py (Model Evaluation)

- Model performance testing
- Metric computation
- Confusion matrix analysis
- Best for model assessment

## Common Operations

### Using main.py

```bash
# Train all models
python main.py --model all

# Train single model
python main.py --model efficientnet --drive True --batch 32
```

### Using train.py

```bash
# Basic training
python train.py --model efficientnet

# Resume training
python train.py --model efficientnet --resume path/to/checkpoint.pth
```

### Configuration

- Model configurations in `config/base_config.py`
- Data paths in `config/paths.py`
- Model-specific parameters through command line arguments
