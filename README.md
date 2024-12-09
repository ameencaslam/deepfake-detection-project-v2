# Deepfake Detection Project

A comprehensive deepfake detection system implementing multiple state-of-the-art architectures.

## Available Models

1. **Swin Transformer** (`swin`): Vision transformer with shifted windows
2. **Two-Stream Network** (`two_stream`): Spatial and frequency domain analysis
3. **Xception** (`xception`): Modified Xception with multi-scale features
4. **CNN-Transformer Hybrid** (`cnn_transformer`): Combined CNN and transformer architecture
5. **Cross-Attention Model** (`cross_attention`): Multi-head cross-attention mechanism
6. **EfficientNet-B3** (`efficientnet`): Optimized CNN with compound scaling

## Usage

```bash
# Train a model
python train.py --model efficientnet --data_path /path/to/data --batch_size 32

# Evaluate a model
python evaluate.py --model efficientnet --checkpoint path/to/checkpoint.pth --data_path /path/to/test_data
```

## Model-Specific Arguments

### EfficientNet

- `--dropout_rate`: Dropout rate for regularization (default: 0.3)
- `--label_smoothing`: Label smoothing factor (default: 0.1)
