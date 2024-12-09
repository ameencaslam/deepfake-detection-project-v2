import argparse
import os
from config.paths import *
from config.base_config import Config
from train import train
from utils.backup import ProjectBackup
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

def train_model(model_name: str, config: Config):
    """Train a single model."""
    logging.info(f"Training model: {model_name}")
    config.model.architecture = model_name
    train(config)

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
    args = parser.parse_args()

    # Setup
    setup_logging()
    
    # Initialize backup system first (this will restore from latest backup if available)
    backup_manager = ProjectBackup(PROJECT_ROOT, use_drive=args.drive)
    
    # Now setup paths and validate dataset (after potential restore)
    setup_paths()
    validate_dataset()

    # Initialize configuration
    config = Config(
        base_path=PROJECT_ROOT,
        use_drive=args.drive
    )

    # Update batch size if provided
    if args.batch != 32:
        config.training.batch_size = args.batch

    # Train models
    if args.model.lower() == 'all':
        for model_name in MODELS.keys():
            if model_name != 'all':
                train_model(model_name, config)
    else:
        train_model(args.model, config)
        
    # Create new backup after training
    backup_manager.create_backup()
    backup_manager.clean_old_backups()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user")
    except Exception as e:
        logging.error(f"\nError during training: {str(e)}")
        raise