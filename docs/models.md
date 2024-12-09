# Model Architectures Documentation

## Overview

The project includes multiple state-of-the-art architectures for deepfake detection.

## Base Model

All models inherit from `BaseModel` which provides:

- Mixed precision training
- Checkpoint management
- Training/validation steps
- Optimizer configuration

## Available Architectures

### 1. Swin Transformer

```python
from models.architectures import get_model
model = get_model('swin_transformer')
```

Features:

- Hierarchical feature learning
- Shifted window attention
- Additional normalization layers
- Stochastic depth dropout

### 2. Two-Stream Network

```python
model = get_model('two_stream')
```

Features:

- Spatial stream (RGB)
- Frequency stream (FFT)
- Feature fusion
- Multi-scale processing

### 3. Xception

```python
model = get_model('xception')
```

Features:

- Depthwise separable convolutions
- Multi-scale feature aggregation
- Additional normalization
- Feature pooling strategies

### 4. CNN-Transformer Hybrid

```python
model = get_model('cnn_transformer')
```

Features:

- CNN backbone (ResNet50)
- Transformer blocks
- Cross-feature attention
- Positional embeddings

### 5. Cross-Attention Model

```python
model = get_model('cross_attention')
```

Features:

- Multi-scale feature extraction
- Cross-attention mechanism
- Global context modeling
- Feature fusion

## Common Features

### Training Optimizations

1. Learning Rate Management:

   - One Cycle Policy
   - Layer-wise learning rates
   - Warmup strategy

2. Regularization:

   - Label smoothing
   - Dropout scheduling
   - Weight decay
   - Layer normalization

3. Feature Processing:
   - Multi-scale features
   - Feature normalization
   - Attention mechanisms

### Model Configuration

```python
model = get_model(
    architecture='swin_transformer',
    pretrained=True,
    num_classes=2,
    dropout_rate=0.3
)
```

## Training Features

1. Mixed Precision:

   ```python
   with torch.cuda.amp.autocast(enabled=self.mixed_precision):
       outputs = model(images)
   ```

2. Checkpointing:

   ```python
   model.save_checkpoint(
       path='checkpoint.pth',
       epoch=epoch,
       optimizer=optimizer
   )
   ```

3. Validation:
   ```python
   metrics = model.validation_step(batch)
   ```

## Important Notes

1. All models support mixed precision training
2. Checkpoints include full training state
3. Models can be customized via configuration
4. Pre-trained weights are available
5. Hardware-specific optimizations are automatic
