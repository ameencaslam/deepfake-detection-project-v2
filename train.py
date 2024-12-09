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
import logging
import numpy as np
from utils.visualization import TrainingVisualizer
import glob
from typing import Optional
from utils.project_manager import ProjectManager

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

def find_latest_checkpoint(model_name: str) -> Optional[str]:
    """Find the latest checkpoint for a model."""
    checkpoint_dir = os.path.join(CHECKPOINTS_PATH, model_name)
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoints:
        return None
        
    # Get latest checkpoint by modification time
    return max(checkpoints, key=os.path.getctime)

def train(config: Config, resume: bool = False):
    """Train a model with the given configuration."""
    try:
        # Initialize project manager
        project_manager = ProjectManager(config.base_path, use_drive=config.use_drive)
        
        # Initialize training controller
        controller = TrainingController()
        
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
        
        # Load checkpoint if resuming
        start_epoch = 0
        if resume:
            checkpoint_path = find_latest_checkpoint(config.model.architecture)
            if checkpoint_path:
                print(f"Resuming from checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=hw_manager.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming from epoch {start_epoch}")
            else:
                print("No checkpoint found, starting from scratch")
        
        # Initialize progress tracking
        progress = ProgressTracker(
            num_epochs=config.training.num_epochs,
            num_batches=len(dataloaders['train']),
            hardware_manager=hw_manager
        )
        
        # Initialize visualizer
        visualizer = TrainingVisualizer(config.paths['results'] / config.model.architecture)
        
        # Plot expected learning rate schedule
        visualizer.plot_learning_rate_schedule(
            optimizer,
            config.training.num_epochs,
            len(dataloaders['train'])
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
            progress_bar = progress.new_epoch(epoch)
            
            for batch_idx, (images, labels) in enumerate(dataloaders['train']):
                # Check if training should stop
                if controller.should_stop():
                    print("\nStopping training as requested...")
                    progress_bar.close()
                    return
                    
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
                
                # Step LR scheduler if it's OneCycleLR
                if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()
                
                # Update progress
                progress.update_batch(
                    batch_idx,
                    metrics['loss'].item(),
                    metrics['accuracy'].item(),
                    progress_bar
                )
            
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
                    batch_metrics = model.validation_step((images, labels))
                    
                    # Accumulate metrics
                    val_metrics['loss'] += batch_metrics['loss'].item() * images.size(0)
                    val_metrics['accuracy'] += batch_metrics['accuracy'].item() * images.size(0)
                    val_metrics['num_samples'] += images.size(0)
                    
                    # Store predictions and labels for plotting
                    val_metrics['all_preds'].extend(batch_metrics['predictions'].cpu().numpy())
                    val_metrics['all_labels'].extend(labels.cpu().numpy())
                    probs = torch.softmax(model(images), dim=1)
                    val_metrics['all_probs'].extend(probs[:, 1].cpu().numpy())
                    
            # Calculate average validation metrics
            val_metrics['loss'] /= val_metrics['num_samples']
            val_metrics['accuracy'] /= val_metrics['num_samples']
            
            # Update ReduceLROnPlateau scheduler if used
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            
            # Update visualizer
            current_lr = optimizer.param_groups[0]['lr']
            visualizer.update_history(
                {
                    'train_loss': progress.running_loss,
                    'train_acc': progress.running_acc,
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy']
                },
                current_lr
            )
            
            # Plot training history
            visualizer.plot_training_history()
            
            # Plot validation metrics every few epochs
            if (epoch + 1) % 5 == 0:
                visualizer.plot_confusion_matrix(
                    np.array(val_metrics['all_labels']),
                    np.array(val_metrics['all_preds'])
                )
                visualizer.plot_roc_curve(
                    np.array(val_metrics['all_labels']),
                    np.array(val_metrics['all_probs'])
                )
                visualizer.plot_prediction_distribution(
                    np.array(val_metrics['all_probs']),
                    np.array(val_metrics['all_labels'])
                )
            
            # End epoch and display metrics
            progress.end_epoch(val_metrics)
            
            # Save checkpoint if best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                checkpoint_dir = os.path.join(
                    config.paths['checkpoints'],
                    config.model.architecture
                )
                best_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_best.pth"
                )
                save_checkpoint(model, best_path, epoch, optimizer, scheduler, val_metrics)
                if config.use_drive:
                    project_manager.backup()  # This will include the new best checkpoint
            
            # Display stop button after each epoch
            print("\nClick 'Stop Training' to stop after this epoch, or let it continue...")
        
        # Save final training summary
        visualizer.save_training_summary(val_metrics, config.model.architecture)
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise
        
    finally:
        # Create final backup
        print("Creating final backup...")
        project_manager.backup()
        project_manager.clean()

def main():
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--base_path', type=str, required=True,
                       help='Base path for saving models and data')
    parser.add_argument('--use_drive', type=bool, default=True,
                       help='Whether to use Google Drive for saving checkpoints')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
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
        train(config, resume=args.resume)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
        
if __name__ == '__main__':
    main() 