# Running on Google Colab

1. Mount Drive
2. Clone repository to `/content/PROJECT-V2`
3. Create symbolic link: `/content/dataset` → `/content/drive/MyDrive/deepfake-project/dataset`
4. Install requirements

## Available Models

- `swin` - Swin Transformer
- `two_stream` - Two-Stream Network
- `xception` - Xception
- `cnn_transformer` - CNN-Transformer Hybrid
- `cross_attention` - Cross-Attention Model
- `efficientnet` - EfficientNet-B3

## Command Options

### Training Options

```
python main.py [OPTIONS]

Options:
--model MODEL_NAME    Model to use (see Available Models above)
--drive BOOL         Use Google Drive (True/False)
--batch INT          Batch size (default: 32)
--resume             Resume from latest checkpoint
```

Examples:

```
# Basic training
python main.py --model efficientnet --drive True

# Resume training
python main.py --model efficientnet --drive True --resume

# Custom batch size
python main.py --model efficientnet --drive True --batch 64
```

### Evaluation Options

```
python evaluate.py [OPTIONS]

Options:
--model MODEL_NAME       Model to evaluate
--checkpoint PATH       Path to specific checkpoint (optional)
--data_path PATH       Path to test data (optional)
--batch_size INT       Batch size for evaluation (default: 32)
```

Examples:

```
# Basic evaluation
python evaluate.py --model efficientnet

# Evaluate specific checkpoint
python evaluate.py --model efficientnet --checkpoint path/to/checkpoint.pth

# Custom test data
python evaluate.py --model efficientnet --data_path /path/to/test/data
```

### Project Management

```
python manage.py [COMMAND]

Commands:
backup              Create project backup
restore            Restore from latest backup
clean              Clean temporary files
```

Examples:

```
# Create backup
python manage.py backup

# Restore from backup
python manage.py restore

# Clean temp files
python manage.py clean
```

## Drive Structure

```
drive/MyDrive/deepfake-project/
├── dataset/           # Your dataset
├── checkpoints/       # Model checkpoints
├── results/          # Training results
└── backups/          # Project backups
```
