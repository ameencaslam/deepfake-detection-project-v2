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
    """Train a model with the given configuration."""
    try:
        # Initialize hardware
        hw_manager = HardwareManager()
        hw_manager.print_hardware_info()
        
        # Initialize visualizer
        results_path = Path(config.paths['results']) / config.model.architecture
        visualizer = TrainingVisualizer(results_path)
        
        # Check for local checkpoint
        local_checkpoint_dir = Path(config.paths['checkpoints']) / config.model.architecture
        local_checkpoint_path = local_checkpoint_dir / "checkpoint_best.pth"
        
        # Debug checkpoint paths
        logging.info(f"Resume flag: {resume}")
        logging.info(f"Looking for checkpoint in local directory only")
        logging.info(f"Local checkpoint path: {local_checkpoint_path}")
        logging.info(f"Local checkpoint exists: {local_checkpoint_path.exists()}")
        if local_checkpoint_dir.exists():
            logging.info(f"Local checkpoint dir contents: {list(local_checkpoint_dir.glob('*'))}")
        
        if resume and local_checkpoint_path.exists():
            # Load checkpoint
            logging.info(f"Loading local checkpoint from: {local_checkpoint_path}")
            checkpoint = torch.load(local_checkpoint_path, map_location=hw_manager.device)
            logging.info(f"Checkpoint contents: {list(checkpoint.keys())}")
            logging.info(f"Previous metrics: {checkpoint['metrics']}")
            
            # Create model without pretrained weights
            logging.info("Creating model architecture without pretrained weights")
            model = get_model(
                config.model.architecture,
                pretrained=False,
                num_classes=config.model.num_classes,
                dropout_rate=config.model.dropout_rate
            )
            
            # Load checkpoint state
            logging.info("Loading model state from checkpoint")
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded model state from checkpoint with validation accuracy: {checkpoint['metrics'].get('accuracy', 0.0):.4f}")
            
            # Move model to device
            logging.info(f"Moving model to device: {hw_manager.device}")
            model = model.to(hw_manager.device)
            
            # Setup optimizer and load its state
            logging.info("Configuring optimizer")
            optimizer = model.configure_optimizers()[0]
            logging.info("Loading optimizer state from checkpoint")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Set training state
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['metrics'].get('accuracy', 0.0)
            logging.info(f"Resuming from epoch {start_epoch} with best validation accuracy: {best_val_acc:.4f}")
            
        else:
            if resume:
                logging.warning(f"Resume requested but no checkpoint found at {local_checkpoint_path}")
            
            # Create new model with pretrained weights
            logging.info("Creating new model with pretrained weights")
            model = get_model(
                config.model.architecture,
                pretrained=config.model.pretrained,
                num_classes=config.model.num_classes,
                dropout_rate=config.model.dropout_rate
            )
            
            # Move model to device
            logging.info(f"Moving model to device: {hw_manager.device}")
            model = model.to(hw_manager.device)
            
            # Setup fresh training state
            logging.info("Initializing fresh training state")
            optimizer = model.configure_optimizers()[0]
            start_epoch = 0
            best_val_acc = 0.0
            
        scheduler = None
        
        # Get dataloaders
        logging.info("Initializing dataloaders")
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
        
        # Initialize progress tracker
        progress = ProgressTracker(
            num_epochs=config.training.num_epochs,
            num_batches=len(dataloaders['train']),
            hardware_manager=hw_manager
        )
        
        # Training loop
        for epoch in range(start_epoch, config.training.num_epochs):
            model.train()
            progress_bar = progress.new_epoch(epoch)
            
            # Training phase
            for batch_idx, batch in enumerate(dataloaders['train']):
                images, labels = batch
                images = images.to(hw_manager.device)
                labels = labels.to(hw_manager.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == labels).float().mean()
                
                # Update progress
                progress.update_batch(
                    batch_idx=batch_idx,
                    loss=loss.item(),
                    accuracy=accuracy.item(),
                    pbar=progress_bar
                )
            
            # End training phase
            progress_bar.close()
            
            # Validation phase
            model.eval()
            val_metrics = {
                'loss': 0.0,
                'accuracy': 0.0,
                'num_samples': 0,
                'all_preds': [],
                'all_labels': [],
                'all_probs': []
            }
            
            with torch.no_grad():
                for images, labels in dataloaders['val']:
                    images = images.to(hw_manager.device)
                    labels = labels.to(hw_manager.device)
                    
                    # Forward pass
                    outputs = model(images)
                    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                    
                    # Get predictions
                    _, predicted = torch.max(outputs.data, 1)
                    accuracy = (predicted == labels).float().mean()
                    
                    # Store results
                    val_metrics['loss'] += loss.item() * images.size(0)
                    val_metrics['accuracy'] += accuracy.item() * images.size(0)
                    val_metrics['num_samples'] += images.size(0)
                    val_metrics['all_preds'].extend(predicted.cpu().numpy())
                    val_metrics['all_labels'].extend(labels.cpu().numpy())
                    probs = torch.softmax(outputs, dim=1)
                    val_metrics['all_probs'].extend(probs[:, 1].cpu().numpy())
            
            # Calculate average metrics
            val_metrics['loss'] /= val_metrics['num_samples']
            val_metrics['accuracy'] /= val_metrics['num_samples']
            
            # End epoch and display metrics
            progress.end_epoch(val_metrics)
            
            # Save checkpoint if best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                checkpoint_dir = Path(config.paths['checkpoints']) / config.model.architecture
                save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=val_metrics,
                    is_best=True
                )
        
        # Save final training summary
        visualizer.save_training_summary(val_metrics, config.model.architecture)
        
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