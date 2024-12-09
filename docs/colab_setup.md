# Colab Setup Guide

## Project Structure in Colab

```
/content/                           # Colab root directory
├── PROJECT-V2/                     # Main project directory
│   ├── config/                     # Configuration files
│   ├── models/                     # Model architectures
│   ├── utils/                     # Utility functions
│   ├── main.py                    # High-level training script
│   ├── train.py                   # Detailed training script
│   └── evaluate.py                # Evaluation script
│
├── dataset/                        # Dataset directory
│   ├── train/                     # Training data
│   │   ├── real/                  # Real images
│   │   └── fake/                  # Fake images
│   └── test/                      # Test data
│       ├── real/                  # Real images
│       └── fake/                  # Fake images
│
└── drive/                         # Google Drive mount point
    └── MyDrive/
        └── deepfake-project/      # Project backups
            ├── checkpoints/       # Model checkpoints
            └── results/           # Training results
```

## Setup Steps

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Clone Repository

```python
!git clone https://github.com/your-repo/PROJECT-V2.git
%cd PROJECT-V2
```

### 3. Install Dependencies

```python
!pip install -r requirements.txt
```

### 4. Set Up Dataset

```python
# Create dataset directories
!mkdir -p /content/dataset/train/real
!mkdir -p /content/dataset/train/fake
!mkdir -p /content/dataset/test/real
!mkdir -p /content/dataset/test/fake

# Copy or download your dataset here
```

## Running Models

### Using main.py (Recommended for Colab)

```python
# Train all models with Drive backup
!python main.py --model all --drive True

# Train single model
!python main.py --model efficientnet --drive True --batch 32
```

### Using train.py (Advanced Control)

```python
# Basic training
!python train.py \
    --model efficientnet \
    --data_path /content/dataset/train \
    --batch_size 32

# Advanced options
!python train.py \
    --model efficientnet \
    --data_path /content/dataset/train \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 50
```

### Evaluation

```python
# Quick evaluation
!python evaluate.py --model efficientnet

# Detailed evaluation
!python evaluate.py \
    --model efficientnet \
    --data_path /content/dataset/test \
    --checkpoint /content/drive/MyDrive/deepfake-project/checkpoints/efficientnet/best.pth
```

## Best Practices for Colab

### 1. Memory Management

- Start with smaller batch sizes (16 or 32)
- Enable GPU runtime (Runtime → Change runtime type → GPU)
- Monitor GPU memory usage
- Clear memory between runs

### 2. Drive Integration

- Always use `--drive True` for backup
- Check Drive space regularly
- Keep important checkpoints in Drive
- Clean old checkpoints periodically

### 3. Dataset Handling

- Keep dataset in /content/dataset
- Use balanced train/test split
- Verify data paths before training
- Monitor data loading speed

### 4. Runtime Settings

```python
# Check GPU availability
!nvidia-smi

# Monitor GPU usage
!nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

## Common Issues and Solutions

### 1. Out of Memory

```python
# Reduce batch size
!python main.py --model efficientnet --batch 16

# Clear GPU memory
import torch
torch.cuda.empty_cache()
```

### 2. Drive Issues

```python
# Remount Drive if disconnected
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### 3. Training Interruption

- Checkpoints are automatically saved to Drive
- Resume training using latest checkpoint
- Use `--resume` flag with train.py

## Getting Help

```python
# View available options
!python main.py --help
!python train.py --help
!python evaluate.py --help
```
