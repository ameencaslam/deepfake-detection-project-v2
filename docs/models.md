# Model Architectures

This document describes the available model architectures for deepfake detection.

## Swin Transformer

A hierarchical vision transformer using shifted windows for efficient attention computation.

**Key Features**:

- Hierarchical feature learning
- Shifted window-based self-attention
- Linear computational complexity
- State-of-the-art performance on vision tasks

## Two-Stream Network

Processes both spatial and frequency domain information for robust deepfake detection.

**Key Features**:

- Parallel spatial and frequency streams
- DCT-based frequency analysis
- Feature fusion module
- Attention-based stream weighting

## Xception

Modified Xception architecture with enhanced feature extraction capabilities.

**Key Features**:

- Depthwise separable convolutions
- Multi-scale feature processing
- Skip connections
- Efficient parameter usage

## CNN-Transformer Hybrid

Combines CNN-based local feature extraction with transformer-based global reasoning.

**Key Features**:

- CNN backbone for local features
- Transformer encoder for global context
- Feature pyramid network
- Multi-head self-attention

## Cross-Attention Model

Utilizes cross-attention mechanisms to capture relationships between real and fake patterns.

**Key Features**:

- Multi-head cross-attention
- Feature matching module
- Dual-path architecture
- Contrastive learning support

## EfficientNet-B3

Optimized CNN architecture with compound scaling for balanced performance.

**Key Features**:

- Compound scaling of depth/width/resolution
- Mobile inverted bottleneck blocks
- Squeeze-and-excitation modules
- Efficient parameter utilization
