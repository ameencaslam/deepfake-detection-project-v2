# Training and Evaluation Guide

## Training Scripts

### main.py (High-Level Interface)

For quick training and multiple model management:

```bash
# Train all models
python main.py --model all

# Train single model with Drive backup
python main.py --model efficientnet --drive True

# Custom batch size
python main.py --model swin --batch 64
```

Available arguments:

- `--model`: Model to train ('all' or specific model name)
- `--drive`: Use Google Drive backup (default: True)
- `--batch`: Batch size (default: 32)

### train.py (Detailed Control)

For fine-grained control over training:

```bash
# Basic training
python train.py --model efficientnet --data_path /path/to/data

# Advanced options
python train.py \
    --model efficientnet \
    --data_path /path/to/data \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 50 \
    --dropout_rate 0.3 \
    --label_smoothing 0.1 \
    --weight_decay 0.01
```

## Script Relationships

- `main.py` calls `train.py` internally
- `main.py` handles project-wide settings
- `train.py` handles detailed training logic
- Both can be used independently

## Training Features

### 1. Optimization

- Mixed precision training
- Layer-wise learning rates
- One-cycle learning rate policy
- Weight decay with AdamW

### 2. Regularization

- Label smoothing
- Dropout
- Layer normalization
- Stochastic depth (model specific)

### 3. Monitoring

- Training loss
- Validation metrics
- Learning rate schedule
- Memory usage

## Evaluation

### Quick Evaluation

```bash
# Evaluate using latest checkpoint
python evaluate.py --model efficientnet

# Evaluate specific checkpoint
python evaluate.py \
    --model efficientnet \
    --checkpoint path/to/checkpoint.pth
```

### Detailed Evaluation

```bash
python evaluate.py \
    --model efficientnet \
    --checkpoint path/to/checkpoint.pth \
    --data_path /path/to/test/data \
    --batch_size 32
```

## Model-Specific Training

### EfficientNet

```bash
python train.py \
    --model efficientnet \
    --dropout_rate 0.3 \
    --label_smoothing 0.1
```

### Swin Transformer

```bash
python train.py \
    --model swin \
    --window_size 7 \
    --num_heads 8
```

## Best Practices

### Using main.py

1. Start with `main.py` for quick experiments
2. Use `--model all` to train all models
3. Enable Drive backup for important runs
4. Adjust batch size if needed

### Using train.py

1. Use for detailed parameter control
2. Monitor validation metrics
3. Use appropriate batch size
4. Enable mixed precision training

### Evaluation

1. Use multiple test sets
2. Check all metrics
3. Analyze confusion matrix
4. Compare with baselines

## Common Issues

### Out of Memory

- Reduce batch size
- Enable mixed precision
- Check input resolution

### Poor Convergence

- Adjust learning rate
- Modify optimizer settings
- Check data preprocessing
- Verify loss computation

## Getting Help

```bash
# View high-level options
python main.py --help

# View detailed training options
python train.py --help

# View evaluation options
python evaluate.py --help
```

## Checkpoints

### Location

- Default: `checkpoints/<model_name>/`
- Latest used automatically
- Specific checkpoint via `--checkpoint`

### Contents

- Model state
- Optimizer state
- Training epoch
- Performance metrics

## Data Management

### Training Data

- Set via `--data_path`
- Default from config
- Supports multiple formats

### Validation/Test

- Automatic split
- Custom test set
- Balanced evaluation
