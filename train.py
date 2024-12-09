"""Training module for deepfake detection models."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from core import load_model
from utils.dataset import DeepfakeDataset
from utils.training import TrainingController
from utils.visualization import TrainingVisualizer
from utils.backup import BackupManager
from config.base_config import Config

def train(config: Config) -> None:
    """Train a model with the given configuration."""
    logger = logging.getLogger('deepfake')
    
    # Setup training components
    model = load_model(config)
    dataset = DeepfakeDataset(config.data)
    training_controller = TrainingController(config.training)
    visualizer = TrainingVisualizer(config.paths.results)
    backup_manager = BackupManager(config.paths.checkpoints)
    
    # Training loop
    for epoch in range(config.training.epochs):
        # Training epoch
        metrics = training_controller.train_epoch(model, dataset)
        
        # Validation
        if epoch % config.training.validate_every == 0:
            val_metrics = training_controller.validate(model, dataset)
            metrics.update(val_metrics)
        
        # Visualization
        visualizer.update(epoch, metrics)
        
        # Checkpointing
        if training_controller.should_save_checkpoint(metrics):
            backup_manager.save_checkpoint(
                model, 
                training_controller.optimizer,
                epoch,
                metrics
            )
            
        # Early stopping
        if training_controller.should_stop(metrics):
            logger.info("Early stopping triggered")
            break
            
    # Final cleanup and visualization
    visualizer.finalize()
    backup_manager.cleanup_old_checkpoints() 