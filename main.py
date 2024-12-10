"""
Main training script for deepfake detection models.

Usage:
    python main.py [OPTIONS]

Options:
    --model MODEL_NAME    Model to train (default: efficientnet)
                         Available models: efficientnet, resnet, vit, all
    --drive BOOL         Use Google Drive for backup (default: True)
    --batch INT          Batch size (default: 32)
    --epochs INT         Number of training epochs (default: from config.py)
    --resume             Resume from latest checkpoint

Examples:
    # Train efficientnet model with default settings
    python main.py --model efficientnet

    # Train with custom batch size and epochs
    python main.py --model efficientnet --batch 64 --epochs 20

    # Resume training from latest checkpoint
    python main.py --model efficientnet --resume

    # Train all available models
    python main.py --model all
"""

import argparse
import os
from config.paths import *
from config.base_config import Config
from train import train
from manage import ProjectManager
import logging
from google.colab import drive

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def setup_drive(use_drive: bool):
    """Setup Google Drive if needed."""
    if use_drive:
        try:
            # Check if drive is already mounted
            if not os.path.exists('/content/drive/MyDrive'):
                drive.mount('/content/drive')
            
            # Verify drive access
            if not os.path.exists('/content/drive/MyDrive'):
                logging.error("Drive mount point exists but MyDrive is not accessible")
                return False
                
            os.makedirs(DRIVE_PATH, exist_ok=True)
            logging.info("Google Drive mounted and verified successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to setup Google Drive: {str(e)}")
            return False
    return False

def train_model(model_name: str, config: Config, resume: bool = False):
    """Train a single model."""
    logging.info(f"Training model: {model_name}")
    config.model.architecture = model_name
    train(config, resume=resume)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train deepfake detection models')
    parser.add_argument('--model', type=str, default='efficientnet',
                       choices=list(MODELS.keys()) + ['all'],
                       help=f'Model to train: {", ".join(list(MODELS.keys()) + ["all"])}')
    parser.add_argument('--drive', type=bool, default=True,
                       help='Use Google Drive for backup')
    parser.add_argument('--batch', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (default: use config value)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')
    args = parser.parse_args()

    # Setup
    setup_logging()
    logging.info(f"Resume training: {args.resume}")
    
    # Setup Drive if needed
    drive_available = False
    if args.drive:
        drive_available = setup_drive(args.drive)
    
    # Initialize project manager
    project_manager = ProjectManager(project_path=PROJECT_ROOT, use_drive=drive_available)
    
    # Now setup paths and validate dataset
    setup_paths()
    validate_dataset()

    # Initialize configuration
    config = Config(
        base_path=PROJECT_ROOT,
        use_drive=drive_available
    )

    # Update batch size if provided
    if args.batch != 32:
        config.training.batch_size = args.batch
        
    # Update number of epochs if provided
    if args.epochs is not None:
        config.training.num_epochs = args.epochs

    # Train models
    if args.model.lower() == 'all':
        for model_name in MODELS.keys():
            if model_name != 'all':
                train_model(model_name, config, resume=args.resume)
    else:
        train_model(args.model, config, resume=args.resume)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user")
    except Exception as e:
        logging.error(f"\nError during training: {str(e)}")
        raise