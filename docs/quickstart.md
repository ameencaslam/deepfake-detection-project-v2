# Quickstart Guide

## Setup in Google Colab

1. Clone and install:

```python
# Clone repository
!git clone https://github.com/ameencaslam/deepfake-detection-project-v2
%cd deepfake-detection-project-v2

# Install dependencies
!pip install -r requirements.txt
```

2. Run training:

```python
# Train a specific model
!python main.py --model swin

# Or train all models
!python main.py --model all

# Custom batch size
!python main.py --model xception --batch 64

# Without Drive backup
!python main.py --model cnn_transformer --drive false
```

## Available Models

- `swin`: Swin Transformer
- `two_stream`: Two-Stream Network
- `xception`: Xception
- `cnn_transformer`: CNN-Transformer Hybrid
- `cross_attention`: Cross-Attention Model
- `all`: Train all models sequentially

## Command Line Arguments

- `--model`: Model to train (default: 'all')
- `--drive`: Use Google Drive backup (default: True)
- `--batch`: Batch size (default: 32)

## Quick Tips

### Memory Management

- Start with default batch size (32)
- Reduce batch size if OOM: `--batch 16`
- Use GPU runtime in Colab

### Training Speed

- Enable GPU in Colab (Runtime → Change runtime type → GPU)
- Default settings are optimized for Colab

### Best Practices

- Monitor training progress
- Use Drive backup when possible
- Start with one model before training all

## Common Issues

### Out of Memory

```python
# Reduce batch size
!python main.py --model swin --batch 16
```

### Drive Issues

```python
# Run without Drive
!python main.py --model swin --drive false
```

### Dataset Not Found

Make sure the dataset is in:

```
/content/3body-filtered-v2-10k/
├── real/
└── fake/
```

## Next Steps

1. Monitor training progress in the output
2. Check saved models in the project directory
3. Results are saved automatically
4. Use Drive backup for important checkpoints
