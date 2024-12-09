import torch
import os
import argparse
from utils.dataset import DeepfakeDataset
from utils.hardware import HardwareManager
from utils.progress import ProgressTracker, TrainingController
from utils.backup import ProjectBackup
from config.base_config import Config
from models.architectures import get_model
from typing import Dict, Any
import json
from datetime import datetime

def save_checkpoint(model, checkpoint_path, epoch, optimizer, scheduler, metrics):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    torch.save(checkpoint, checkpoint_path)

def train(config: Config):
    # Initialize backup system
    backup_manager = ProjectBackup(config.base_path, config.use_drive)
    
    # Initialize training controller
    controller = TrainingController()
    
    try:
        # Initialize hardware
        hw_manager = HardwareManager()
        hw_manager.print_hardware_info()
        
        # Create dataloaders
        dataloaders = DeepfakeDataset.create_dataloaders(
            data_dir=config.paths['data'],
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            image_size=config.data.image_size
        )
        
        # Initialize model
        model = get_model(
            architecture=config.model.architecture,
            pretrained=config.model.pretrained,
            num_classes=config.model.num_classes
        ).to(hw_manager.device)
        
        # Get optimizer and scheduler
        optimizer, scheduler = model.configure_optimizers()
        
        # Initialize progress tracking
        progress = ProgressTracker(
            num_epochs=config.training.num_epochs,
            num_batches=len(dataloaders['train']),
            hardware_manager=hw_manager
        )
        
        # Training loop
        best_val_acc = 0.0
        for epoch in range(config.training.num_epochs):
            # Check if training should stop
            if controller.should_stop():
                print("\nStopping training as requested...")
                break
                
            # Training phase
            model.train()
            pbar = progress.new_epoch(epoch)
            
            for batch_idx, (images, labels) in enumerate(dataloaders['train']):
                images = images.to(hw_manager.device)
                labels = labels.to(hw_manager.device)
                
                # Training step
                metrics = model.training_step((images, labels))
                
                # Optimizer step with gradient scaling for mixed precision
                optimizer.zero_grad()
                if hasattr(model, 'scaler') and model.scaler:
                    model.scaler.scale(metrics['loss']).backward()
                    model.scaler.step(optimizer)
                    model.scaler.update()
                else:
                    metrics['loss'].backward()
                    optimizer.step()
                    
                # Update progress
                progress.update_batch(
                    batch_idx,
                    metrics['loss'].item(),
                    metrics['accuracy'].item(),
                    pbar
                )
                
            # Validation phase
            model.eval()
            val_metrics = {
                'loss': 0.0,
                'accuracy': 0.0,
                'num_samples': 0
            }
            
            with torch.no_grad():
                for images, labels in dataloaders['val']:
                    images = images.to(hw_manager.device)
                    labels = labels.to(hw_manager.device)
                    batch_metrics = model.validation_step((images, labels))
                    val_metrics['loss'] += batch_metrics['loss'].item() * images.size(0)
                    val_metrics['accuracy'] += batch_metrics['accuracy'].item() * images.size(0)
                    val_metrics['num_samples'] += images.size(0)
                    
            # Calculate average validation metrics
            val_metrics['loss'] /= val_metrics['num_samples']
            val_metrics['accuracy'] /= val_metrics['num_samples']
            
            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
                
            # End epoch and display metrics
            progress.end_epoch(val_metrics)
            
            # Display stop button after each epoch
            print("\nClick 'Stop Training' to stop after this epoch, or let it continue...")
            
            # Save checkpoint
            checkpoint_dir = os.path.join(
                config.paths['checkpoints'],
                config.model.architecture
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save latest checkpoint
            latest_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_latest.pth"
            )
            save_checkpoint(model, latest_path, epoch, optimizer, scheduler, val_metrics)
            
            # Save best checkpoint if needed
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_best.pth"
                )
                save_checkpoint(model, best_path, epoch, optimizer, scheduler, val_metrics)
                backup_manager.backup_to_drive(
                    best_path, 
                    'checkpoints',
                    model_name=config.model.architecture
                )
                
            # Memory optimization
            if hasattr(hw_manager, 'optimize_memory'):
                hw_manager.optimize_memory()
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
        
    finally:
        # Create final backup
        print("Creating final backup...")
        backup_manager.create_backup(include_checkpoints=True)
        backup_manager.clean_old_backups(keep_last=3)

def main():
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--base_path', type=str, required=True,
                       help='Base path for saving models and data')
    parser.add_argument('--use_drive', type=bool, default=True,
                       help='Whether to use Google Drive for saving checkpoints')
    args = parser.parse_args()
    
    # Initialize config
    config = Config(base_path=args.base_path, use_drive=args.use_drive)
    
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
        train(config)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
        
if __name__ == '__main__':
    main() 