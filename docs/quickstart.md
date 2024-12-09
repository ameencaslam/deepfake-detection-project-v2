# Quickstart Guide

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Basic Usage

The project uses a simple command-line interface through `run.py`:

### Training

```bash
# Train with default settings (EfficientNet)
python run.py train

# Train specific model
python run.py train --model swin

# Train with custom parameters
python run.py train --model xception --batch 64 --epochs 200 --lr 0.001
```

### Evaluation

```bash
# Evaluate a trained model
python run.py evaluate --model swin --checkpoint path/to/checkpoint.pth
```

### Hardware Options

```bash
# Use CPU
python run.py train --device cpu

# Use GPU with custom memory fraction
python run.py train --device cuda --gpu-mem-frac 0.8
```

## Dataset Structure

Place your dataset in the following structure:

```
data/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

## Available Models

- `swin`: Swin Transformer
- `two_stream`: Two-Stream Network
- `xception`: Xception
- `cnn_transformer`: CNN-Transformer Hybrid
- `cross_attention`: Cross-Attention Model
- `efficientnet`: EfficientNet-B3 (default)
