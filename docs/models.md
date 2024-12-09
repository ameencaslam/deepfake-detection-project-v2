# Models Documentation

## Available Models

### 1. Swin Transformer (`swin`)

- Vision transformer with shifted windows
- Hierarchical feature learning
- Multi-head self-attention

### 2. Two-Stream Network (`two_stream`)

- Spatial and frequency domain analysis
- Dual-path architecture
- Feature fusion mechanism

### 3. Xception (`xception`)

- Modified Xception architecture
- Multi-scale feature extraction
- Depthwise separable convolutions

### 4. CNN-Transformer Hybrid (`cnn_transformer`)

- Combined CNN and transformer architecture
- Feature pyramid network
- Cross-attention mechanism

### 5. Cross-Attention Model (`cross_attention`)

- Multi-head cross-attention
- Feature interaction learning
- Hierarchical representation

### 6. EfficientNet-B3 (`efficientnet`)

- Compound scaling for optimal performance
- Multi-scale feature aggregation
- Advanced regularization techniques

## Model Usage

### Basic Training

```bash
python train.py --model MODEL_NAME
```

### Custom Training

```bash
python train.py \
    --model MODEL_NAME \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 50
```

### Evaluation

```bash
# Quick evaluation
python evaluate.py --model MODEL_NAME

# Custom evaluation
python evaluate.py \
    --model MODEL_NAME \
    --checkpoint path/to/checkpoint.pth
```

## Model-Specific Parameters

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

### Two-Stream Network

```bash
python train.py \
    --model two_stream \
    --fusion_type 'concat'
```

## Model Features

### Common Features

- Mixed precision training
- Label smoothing
- Layer-wise learning rates
- Automatic checkpoint management

### Advanced Features

- Multi-scale feature processing
- Adaptive learning rate scheduling
- Regularization techniques
- Confusion matrix evaluation

## Model Performance

Each model provides:

- Accuracy metrics
- Precision and recall
- F1 score
- AUC-ROC curve
- Confusion matrix

## Best Practices

1. **Training**

   - Start with default parameters
   - Use mixed precision training
   - Monitor validation metrics
   - Use appropriate batch size

2. **Evaluation**

   - Use multiple metrics
   - Check confusion matrix
   - Validate on different datasets
   - Compare model variants

3. **Optimization**
   - Adjust learning rates
   - Tune dropout rates
   - Modify batch sizes
   - Experiment with schedulers
