# Utilities Documentation

## Overview

The utilities package provides essential tools for training, data management, and hardware optimization.

## Components

### 1. Hardware Management (`utils/hardware.py`)

```python
from utils.hardware import HardwareManager

hw_manager = HardwareManager()
hw_manager.print_hardware_info()
```

Features:

- Automatic device detection
- Memory optimization
- Hardware monitoring
- Mixed precision setup

### 2. Progress Tracking (`utils/progress.py`)

```python
from utils.progress import ProgressTracker

tracker = ProgressTracker(num_epochs=50, num_batches=100)
```

Features:

- Training progress bars
- Metric tracking
- Time estimation
- Hardware usage display

### 3. Dataset Management (`utils/dataset.py`)

```python
from utils.dataset import DeepfakeDataset

dataset = DeepfakeDataset(
    data_dir='path/to/data',
    split='train'
)
```

Features:

- Automatic data splitting
- Data augmentation
- Balanced sampling
- Multi-worker loading

### 4. Training Utilities (`utils/training.py`)

```python
from utils.training import get_optimizer, get_scheduler

optimizer = get_optimizer(model, optimizer_name='adam')
scheduler = get_scheduler(optimizer, scheduler_name='onecycle')
```

Features:

- Optimizer configuration
- Learning rate scheduling
- Label smoothing
- Early stopping

### 5. Backup System (`utils/backup.py`)

```python
from utils.backup import ProjectBackup

backup = ProjectBackup(base_path='/path/to/project')
backup.create_backup()
```

Features:

- Automatic project backup
- Google Drive integration
- Checkpoint management
- Project restoration

## Usage Examples

### 1. Hardware Optimization

```python
# Memory management
hw_manager.optimize_memory()

# Get hardware stats
stats = hw_manager.get_hardware_stats()
print(f"GPU Usage: {stats.gpu_utilization}%")
```

### 2. Progress Tracking

```python
# Start epoch
pbar = tracker.new_epoch(epoch)

# Update batch progress
tracker.update_batch(batch_idx, loss, accuracy, pbar)

# End epoch
tracker.end_epoch(val_metrics)
```

### 3. Dataset Creation

```python
# Create dataloaders
dataloaders = DeepfakeDataset.create_dataloaders(
    data_dir='path/to/data',
    batch_size=32,
    num_workers=4
)
```

### 4. Training Setup

```python
# Configure training
optimizer = get_optimizer(
    model,
    optimizer_name='adamw',
    learning_rate=1e-4
)

scheduler = get_scheduler(
    optimizer,
    scheduler_name='onecycle',
    num_epochs=50
)
```

### 5. Backup Management

```python
# Create backup
backup.create_backup(include_checkpoints=True)

# Restore from backup
backup.restore_from_backup()

# Clean old backups
backup.clean_old_backups(keep_last=3)
```

## Important Notes

1. Hardware manager automatically selects best device
2. Progress tracker provides real-time updates
3. Dataset handles data splitting automatically
4. Training utilities support mixed precision
5. Backup system integrates with Google Drive
6. All components work together seamlessly
