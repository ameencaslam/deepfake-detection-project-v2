# Training and Evaluation Guide

## Training

### Basic Usage

```bash
# Train a model
python train.py --model efficientnet --data_path /path/to/data

# Resume training
python train.py --model efficientnet --resume path/to/checkpoint.pth
```

### Advanced Options

```bash
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

## Evaluation Metrics

### 1. Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

### 2. Detailed Analysis

- Confusion matrix
- True/False positives
- True/False negatives
- Total samples processed

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

### Training

1. Start with default hyperparameters
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
# View training options
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
