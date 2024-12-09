# Deepfake Detection Project

A comprehensive deepfake detection system supporting various model architectures.

## Project Structure

```
project/
├── config/                 # Configuration files
├── models/                 # Model architectures
├── utils/                 # Utility functions
├── docs/                  # Documentation
├── main.py               # Main training script
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── manage.py             # Project management
└── requirements.txt      # Dependencies
```

## Features

- Support for various model architectures
- Optimized training pipeline
- Unified project management system
- Google Drive integration
- Real-time progress tracking
- Hardware optimization
- Automatic checkpoint management
- Training visualization
- Comprehensive evaluation metrics

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare dataset:

   - Place images in `dataset/` directory
   - Dataset split info will be saved automatically

3. Start training (example using EfficientNet):

   ```bash
   python main.py --model efficientnet --drive True
   ```

4. Evaluate model:
   ```bash
   python evaluate.py --model efficientnet
   ```

## Project Management

The `manage.py` script provides unified project management:

```python
from manage import ProjectManager

manager = ProjectManager(project_path='project', use_drive=True)
manager.backup()    # Create backup
manager.restore()   # Restore from backup
manager.clean()     # Clean temp files
```

## Documentation

- `quickstart.md` - Getting started guide
- `models.md` - Model architectures and integration
- `config.md` - Configuration system
- `colab_setup.md` - Google Colab setup

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- Google Drive for backup (optional)

## License

MIT License
