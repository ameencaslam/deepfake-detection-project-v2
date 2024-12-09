# Utilities Documentation

## Overview

The project includes various utility modules for training, evaluation, and project management.

## Hardware Management (`utils/hardware.py`)

```python
from utils.hardware import get_device, get_memory_usage

# Get appropriate device
device = get_device()  # Returns 'cuda' if available, else 'cpu'

# Monitor memory
memory_stats = get_memory_usage()
```

Features:

- Automatic device selection
- Memory monitoring
- GPU optimization
- Resource tracking

## Dataset Management (`utils/dataset.py`)

```python
from utils.dataset import get_dataloader

# Get training dataloader
train_loader = get_dataloader(
    data_path='/path/to/data',
    batch_size=32,
    train=True
)

# Get test dataloader
test_loader = get_dataloader(
    data_path='/path/to/test',
    batch_size=32,
    train=False
)
```

Features:

- Data loading
- Augmentation
- Preprocessing
- Batch management

## Training Utilities (`utils/training.py`)

```python
from utils.training import (
    get_optimizer,
    get_scheduler,
    LabelSmoothingLoss
)

# Setup training components
optimizer = get_optimizer(model, 'adamw', learning_rate=1e-4)
scheduler = get_scheduler(optimizer, 'onecycle', num_epochs=50)
criterion = LabelSmoothingLoss(num_classes=2, smoothing=0.1)
```

Features:

- Optimizer configuration
- Learning rate scheduling
- Loss functions
- Training helpers

## Progress Tracking (`utils/progress.py`)

```python
from utils.progress import ProgressTracker

tracker = ProgressTracker(
    num_epochs=50,
    steps_per_epoch=100
)

# Update progress
tracker.update(loss=0.5, accuracy=0.95)
```

Features:

- Training progress
- Metric tracking
- Time estimation
- Status display

## Backup System (`utils/backup.py`)

```python
from utils.backup import create_backup, restore_backup

# Create backup
create_backup(
    source_dir='project',
    backup_dir='backups'
)

# Restore from backup
restore_backup(
    backup_path='backups/latest',
    restore_dir='project'
)
```

Features:

- Project backups
- State preservation
- Easy restoration
- Version control

## Common Functions

### 1. Model Management

```python
# Get model
model = get_model('efficientnet')

# Load checkpoint
load_checkpoint(model, 'path/to/checkpoint.pth')

# Save checkpoint
save_checkpoint(model, 'save/path.pth')
```

### 2. Training Helpers

```python
# Configure training
optimizer = get_optimizer(model)
scheduler = get_scheduler(optimizer)
criterion = get_loss_function()

# Training step
loss = training_step(model, batch, criterion)
```

### 3. Evaluation Helpers

```python
# Evaluate model
metrics = evaluate_model(model, test_loader)

# Print metrics
print_metrics(metrics)
```

## Best Practices

### 1. Hardware Usage

- Monitor memory usage
- Use appropriate batch sizes
- Enable mixed precision
- Track resource usage

### 2. Data Management

- Verify data paths
- Check data loading
- Monitor batch sizes
- Use proper transforms

### 3. Training Management

- Track progress
- Save checkpoints
- Monitor metrics
- Handle interruptions

## Error Handling

```python
try:
    # Training code
    train_model()
except RuntimeError as e:
    handle_training_error(e)
finally:
    cleanup_resources()
```

## Configuration

```python
# Load config
config = load_config()

# Update settings
config.update({
    'batch_size': 32,
    'learning_rate': 1e-4
})
```

## Logging

```python
# Setup logging
logger = setup_logger()

# Log events
logger.info('Training started')
logger.error('Error occurred')
```
