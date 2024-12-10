"""
Main training script for deepfake detection models.

Usage:
    python main.py [OPTIONS]

Options:
    --model MODEL       Model architecture (default: xception)
                       Available: xception, cnn_transformer, cross_attention, two_stream
    --epochs INT       Number of training epochs (default: 50)
    --batch INT        Batch size (default: 32)
    --lr FLOAT        Learning rate (default: 1e-4)
    --resume          Resume from latest checkpoint
    --data PATH       Path to dataset (default: ./data)
    
Model-specific options:
    --hidden_dim INT   Hidden dimension size (default: 512)
    --num_heads INT    Number of attention heads (default: 8)
    --num_layers INT   Number of transformer layers (default: 3)
    --no_attention    Disable attention mechanism (for supported models)

Example:
    # Train Xception model
    python main.py --model xception --epochs 50 --batch 32
    
    # Train CNN-Transformer with custom settings
    python main.py --model cnn_transformer --hidden_dim 768 --num_heads 12
"""

import argparse
import os
import logging
from pathlib import Path
from config.base_config import Config
from train import train

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    
    # Basic options
    parser.add_argument('--model', type=str, default='xception',
                      choices=['xception', 'cnn_transformer', 'cross_attention', 'two_stream'],
                      help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--resume', action='store_true',
                      help='Resume from latest checkpoint')
    parser.add_argument('--data', type=str, default='./data',
                      help='Path to dataset')
    
    # Model-specific options
    parser.add_argument('--hidden_dim', type=int, default=512,
                      help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                      help='Number of transformer layers')
    parser.add_argument('--no_attention', action='store_true',
                      help='Disable attention mechanism')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    
    # Initialize config
    config = Config(base_path=os.path.dirname(os.path.abspath(__file__)))
    
    # Update config with command line arguments
    config.model.architecture = args.model
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch
    config.training.learning_rate = args.lr
    config.paths['data'] = args.data
    
    # Model-specific settings
    config.model.hidden_dim = args.hidden_dim
    config.model.num_heads = args.num_heads
    config.model.num_layers = args.num_layers
    config.model.use_attention = not args.no_attention
    
    # Start training
    try:
        train(config, resume=args.resume)
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user")
    except Exception as e:
        logging.error(f"\nError during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()