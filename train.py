"""
Training module for deepfake detection models.

This module can be used directly or through main.py. It provides more detailed
control over the training process when used directly.

Usage:
    python train.py [OPTIONS]

Options:
    --base_path PATH     Base path for project (required)
    --use_drive BOOL     Use Google Drive for backup (default: True)
    --epochs INT         Number of training epochs (default: from config.py)
    --resume             Resume from latest checkpoint

Examples:
    # Train with default settings
    python train.py --base_path /path/to/project

    # Train with custom epochs and resume
    python train.py --base_path /path/to/project --epochs 20 --resume

Note:
    For most use cases, it's recommended to use main.py instead, which provides
    a simpler interface and handles model selection.
"""

import torch
import os
import argparse
from pathlib import Path
from utils.dataset import DeepfakeDataset
from utils.hardware import HardwareManager
from utils.progress import ProgressTracker, TrainingController
from manage import ProjectManager
from config.base_config import Config
from models.architectures import get_model
from typing import Dict, Any
import json
from datetime import datetime
import logging
import numpy as np
from utils.visualization import TrainingVisualizer
import glob
from typing import Optional

def save_checkpoint(model, checkpoint_dir, epoch, optimizer, scheduler, metrics, is_best=False):
    """Save model checkpoint.
    
    Args:
        model: Model to save
        checkpoint_dir: Directory to save checkpoint
        epoch: Current epoch
        optimizer: Optimizer state
        scheduler: Scheduler state
        metrics: Training metrics
        is_best: Whether this is the best model so far
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "checkpoint_best.pth"
        torch.save(checkpoint, best_path)
        logging.info(f"Saved best checkpoint with accuracy: {metrics['accuracy']:.4f}")
        
        # Remove any other checkpoints
        for ckpt in checkpoint_dir.glob("checkpoint_*.pth"):
            if ckpt != best_path:
                ckpt.unlink()
                logging.info(f"Removed old checkpoint: {ckpt}")

def train(config: Config, resume: bool = False):
    """Train a model with the given configuration.
    
    Args:
        config: Training configuration
        resume: Whether to resume from checkpoint
    """
    try:
        # Initialize hardware and paths
        hw_manager = HardwareManager()
        checkpoint_dir = Path(config.paths['checkpoints']) / config.model.architecture
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / 'checkpoint_best.pth'
        
        # Initialize visualizer
        visualizer = TrainingVisualizer(Path(config.paths['results']) / config.model.architecture)
        
        # Create model
        model = get_model(
            config.model.architecture,
            pretrained=config.model.pretrained,
            num_classes=config.model.num_classes,
            dropout_rate=config.model.dropout_rate,
            label_smoothing=config.model.label_smoothing,
            hidden_dim=getattr(config.model, 'hidden_dim', 512),
            num_heads=getattr(config.model, 'num_heads', 8),
            num_layers=getattr(config.model, 'num_layers', 3),
            use_attention=getattr(config.model, 'use_attention', True)
        )
        model = model.to(hw_manager.device)
        
        # Configure optimizers
        optimizer, scheduler = model.configure_optimizers()
        
        # Load checkpoint if resuming
        start_epoch = 0
        best_val_loss = float('inf')
        if resume and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['metrics']['loss']
            logging.info(f"Resuming from checkpoint with validation loss: {best_val_loss:.4f}")
        
        # Get dataloaders
        dataloaders = {
            'train': DeepfakeDataset.get_dataloader(
                data_path=config.paths['data'],
                batch_size=config.training.batch_size,
                num_workers=config.data.num_workers,
                image_size=config.data.image_size,
                train=True
            ),
            'val': DeepfakeDataset.get_dataloader(
                data_path=config.paths['data'],
                batch_size=config.training.batch_size,
                num_workers=config.data.num_workers,
                image_size=config.data.image_size,
                train=False
            )
        }
        
        # Initialize progress tracker and controller
        progress = ProgressTracker(
            num_epochs=config.training.num_epochs,
            num_batches=len(dataloaders['train']),
            hardware_manager=hw_manager
        )
        controller = TrainingController()
        
        # Training loop
        for epoch in range(start_epoch, config.training.num_epochs):
            # Check for stop signal
            if controller.should_stop():
                logging.info("Training stopped by user")
                break
            
            # Training phase
            model.train()
            train_metrics = {}
            progress_bar = progress.new_epoch(epoch)
            
            for batch_idx, batch in enumerate(dataloaders['train']):
                # Training step
                batch_metrics = model.training_step(batch)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                batch_metrics['loss'].backward()
                optimizer.step()
                
                # Update metrics
                for key in ['loss', 'accuracy']:
                    if key not in train_metrics:
                        train_metrics[key] = 0.0
                    train_metrics[key] += batch_metrics[key].item()
                
                # Update progress
                progress_bar.update(1)
            
            # Average training metrics
            num_batches = len(dataloaders['train'])
            for key in train_metrics:
                train_metrics[key] /= num_batches
            
            # Validation phase
            model.eval()
            val_metrics = {}
            
            with torch.no_grad():
                for batch in dataloaders['val']:
                    # Validation step
                    batch_metrics = model.validation_step(batch)
                    
                    # Update metrics
                    for key in ['loss', 'accuracy']:
                        if key not in val_metrics:
                            val_metrics[key] = 0.0
                        val_metrics[key] += batch_metrics[key].item()
            
            # Average validation metrics
            num_batches = len(dataloaders['val'])
            for key in val_metrics:
                val_metrics[key] /= num_batches
            
            # Update progress and check if best model
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
            
            # Save checkpoint if validation loss improved
            if is_best:
                save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=val_metrics,
                    is_best=True
                )
                logging.info(f"Saved new best model with validation loss: {val_metrics['loss']:.4f}")
            
            # Update visualizations
            visualizer.update_training_plots(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics
            )
            
            # Step scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Log epoch summary
            logging.info(
                f"Epoch {epoch}/{config.training.num_epochs-1} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
    
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--base_path', type=str, required=True,
                       help='Base path for saving models and data')
    parser.add_argument('--use_drive', type=bool, default=True,
                       help='Whether to use Google Drive for saving checkpoints')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (default: use config value)')
    args = parser.parse_args()
    
    # Initialize config
    config = Config(base_path=args.base_path, use_drive=args.use_drive)
    
    # Update number of epochs if provided
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{config.model.architecture}_{timestamp}"
    experiment_dir = os.path.join(config.paths['results'], experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=4, default=str)
        
    # Start training
    try:
        train(config, resume=args.resume)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
        
if __name__ == '__main__':
    main() 