"""Configuration management utilities."""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import argparse
from config.base_config import Config

AVAILABLE_MODELS = {
    'swin': 'Swin Transformer',
    'two_stream': 'Two-Stream Network',
    'xception': 'Xception',
    'cnn_transformer': 'CNN-Transformer Hybrid',
    'cross_attention': 'Cross-Attention Model',
    'efficientnet': 'EfficientNet-B3'
}

def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file and/or defaults."""
    config = Config()
    
    if config_path:
        with open(config_path) as f:
            file_config = yaml.safe_load(f)
            config.update(file_config)
    
    return config

def get_cli_parser() -> argparse.ArgumentParser:
    """Get the command line argument parser."""
    parser = argparse.ArgumentParser(description='Deepfake Detection System')
    
    # Core arguments
    parser.add_argument('action', choices=['train', 'evaluate'],
                       help='Action to perform')
    parser.add_argument('--model', type=str, default='efficientnet',
                       choices=list(AVAILABLE_MODELS.keys()),
                       help=f'Model to use: {", ".join(AVAILABLE_MODELS.keys())}')
    
    # Training arguments
    parser.add_argument('--batch', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    # Evaluation arguments
    parser.add_argument('--checkpoint', type=str,
                       help='Checkpoint path for evaluation')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--gpu-mem-frac', type=float, default=0.9,
                       help='GPU memory fraction to use (default: 0.9)')
    
    return parser

def merge_cli_config(config: Config, args: argparse.Namespace) -> Config:
    """Merge command line arguments into config."""
    # Update model config
    config.model.architecture = args.model
    
    # Update training config
    config.training.batch_size = args.batch
    if hasattr(args, 'epochs'):
        config.training.epochs = args.epochs
    if hasattr(args, 'lr'):
        config.training.learning_rate = args.lr
    
    # Update hardware config
    config.hardware.device = args.device
    config.hardware.gpu_memory_fraction = args.gpu_mem_frac
    
    # Update evaluation config
    if hasattr(args, 'checkpoint'):
        config.evaluation.checkpoint_path = args.checkpoint
    
    return config 