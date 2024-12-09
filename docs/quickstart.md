# Quickstart Guide

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Available Models

- Swin Transformer (`swin`)
- Two-Stream Network (`two_stream`)
- Xception (`xception`)
- CNN-Transformer Hybrid (`cnn_transformer`)
- Cross-Attention Model (`cross_attention`)
- EfficientNet-B3 (`efficientnet`)

## Training

Basic training command:

```bash
python train.py --model efficientnet --data_path /path/to/data
```

With custom parameters:

```bash
python train.py \
    --model efficientnet \
    --data_path /path/to/data \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 50 \
    --dropout_rate 0.3 \
    --label_smoothing 0.1
```

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
# Training options
python train.py --help

# Evaluation options
python evaluate.py --help

# Main script options
python main.py --help
```

## Project Structure

```
project/
├── config/             # Configuration files
├── models/             # Model architectures
│   └── architectures/  # Individual model implementations
├── utils/             # Utility functions
├── docs/              # Documentation
├── train.py          # Training script
├── evaluate.py       # Evaluation script
└── requirements.txt  # Project dependencies
```

## Common Operations

### Model Training

```bash
# Basic training
python train.py --model efficientnet

# Resume training
python train.py --model efficientnet --resume path/to/checkpoint.pth
```

### Model Evaluation

```bash
# Quick evaluation with latest checkpoint
python evaluate.py --model efficientnet

# Detailed evaluation with specific checkpoint
python evaluate.py --model efficientnet --checkpoint path/to/checkpoint.pth
```

### Configuration

- Model configurations in `config/base_config.py`
- Data paths in `config/paths.py`
- Model-specific parameters through command line arguments
