# Training Guide

This document describes the training process and available options.

## Basic Training

The simplest way to train a model:

```bash
python run.py train
```

This will use the default settings:

- Model: EfficientNet-B3
- Batch size: 32
- Learning rate: 0.001
- Epochs: 100
- Device: CUDA (if available)

## Advanced Options

### Model Selection

```bash
python run.py train --model swin
```

Available models:

- `swin`: Swin Transformer
- `two_stream`: Two-Stream Network
- `xception`: Xception
- `cnn_transformer`: CNN-Transformer Hybrid
- `cross_attention`: Cross-Attention Model
- `efficientnet`: EfficientNet-B3

### Training Parameters

```bash
python run.py train \
    --model xception \
    --batch 64 \
    --epochs 200 \
    --lr 0.001
```

### Hardware Options

```bash
# Use CPU
python run.py train --device cpu

# Use GPU with memory limit
python run.py train --device cuda --gpu-mem-frac 0.8
```

## Training Features

1. **Automatic Checkpointing**

   - Best model saved automatically
   - Regular interval backups
   - Checkpoint naming includes metrics

2. **Early Stopping**

   - Monitors validation loss
   - Prevents overfitting
   - Configurable patience

3. **Mixed Precision Training**

   - Automatic mixed precision
   - Faster training
   - Lower memory usage

4. **Visualization**
   - Training metrics
   - Learning rate schedule
   - Confusion matrices
   - Feature visualizations

## Best Practices

1. **Data Preparation**

   - Use balanced dataset
   - Apply appropriate augmentations
   - Verify data quality

2. **Model Selection**

   - EfficientNet: Good baseline
   - Swin: Better accuracy
   - CNN-Transformer: Best accuracy but slower

3. **Training Tips**
   - Start with default parameters
   - Monitor validation metrics
   - Use appropriate batch size
   - Enable mixed precision
