# Deepfake Detection Project

A comprehensive deepfake detection system with multiple architectures and training optimizations.

## Project Structure

```
deepfake_detection/
├── config/                 # Configuration files
├── models/                 # Model architectures
│   └── architectures/     # Different model implementations
├── utils/                 # Utility functions
├── docs/                  # Documentation
└── train.py              # Main training script
```

## Features

- Multiple state-of-the-art architectures
- Optimized training pipeline
- Automatic backup system
- Google Drive integration
- Progress tracking
- Hardware optimization

## Architectures

1. Swin Transformer
2. Two-Stream Network
3. Xception
4. CNN-Transformer Hybrid
5. Cross-Attention Model

## Getting Started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your dataset:

   - Place real images in `data/real/`
   - Place fake images in `data/fake/`

3. Start training:

   ```python
   from config.base_config import Config
   from train import train

   config = Config(base_path='/path/to/project')
   train(config)
   ```

## Documentation

- See individual README files in the `docs/` folder for detailed documentation of each component
- Each architecture has its own documentation with specific features and configurations

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA compatible GPU (recommended)
- Google Drive for backup (optional)

## Features

- Automatic dataset splitting
- Mixed precision training
- Learning rate scheduling
- Progress tracking
- Model checkpointing
- Automatic backups

## License

MIT License
