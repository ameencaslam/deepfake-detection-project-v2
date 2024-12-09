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
            drive.mount('/content/drive')
            os.makedirs(DRIVE_PATH, exist_ok=True)
            logging.info("Google Drive mounted successfully")
        except:
            logging.warning("Failed to mount Google Drive. Continuing without it.")
            return False
    return use_drive

def train_model(model_name: str, config: Config):
    """Train a single model."""
    logging.info(f"Training model: {model_name}")
    config.model.architecture = model_name
    train(config)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train deepfake detection models')
    parser.add_argument('--model', type=str, default='all',
                       help='Model to train (swin/two_stream/xception/cnn_transformer/cross_attention/all)')
    parser.add_argument('--drive', type=bool, default=True,
                       help='Use Google Drive for backup')
    parser.add_argument('--batch', type=int, default=32,
                       help='Batch size')
    args = parser.parse_args()

    # Setup
    setup_logging()
    setup_paths()
    validate_dataset()
    use_drive = setup_drive(args.drive)

    # Initialize configuration
    config = Config(
        base_path=PROJECT_ROOT,
        use_drive=use_drive
    )

    # Update batch size if provided
    if args.batch != 32:
        config.training.batch_size = args.batch

    # Train models
    if args.model.lower() == 'all':
        for model_key in MODELS:
            train_model(MODELS[model_key], config)
    else:
        model_name = get_model_name(args.model.lower())
        train_model(model_name, config)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user")
    except Exception as e:
        logging.error(f"\nError during training: {str(e)}")
        raise 