import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from torch.cuda.amp import GradScaler
import logging

class BaseModel(nn.Module, ABC):
    """Base model class that all models should inherit from."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.device = config['device']
        self.mixed_precision = config.get('mixed_precision', True)
        self.scaler = GradScaler('cuda') if self.mixed_precision else None
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass
        
    @abstractmethod
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        """Configure optimizer and scheduler for the model.
        
        Returns:
            Tuple of (optimizer, scheduler)
        """
        pass
        
    @abstractmethod
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single training step.
        
        Args:
            batch: Tuple of (inputs, targets)
            
        Returns:
            Dict containing at minimum {'loss': loss_tensor}
        """
        pass
        
    @abstractmethod
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single validation step.
        
        Args:
            batch: Tuple of (inputs, targets)
            
        Returns:
            Dict containing validation metrics
        """
        pass
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, 
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Model optimizer
            
        Returns:
            Dict of epoch metrics
        """
        self.train()
        epoch_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'num_samples': 0
        }
        
        for batch_idx, batch in enumerate(train_loader):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                metrics = self.training_step(batch)
            
            # Backward pass with gradient scaling
            if self.mixed_precision:
                self.scaler.scale(metrics['loss']).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                metrics['loss'].backward()
                optimizer.step()
            
            # Update epoch metrics
            batch_size = batch[0].size(0)
            epoch_metrics['loss'] += metrics['loss'].item() * batch_size
            epoch_metrics['accuracy'] += metrics['accuracy'].item() * batch_size
            epoch_metrics['num_samples'] += batch_size
        
        # Calculate averages
        for key in ['loss', 'accuracy']:
            epoch_metrics[key] /= epoch_metrics['num_samples']
            
        return epoch_metrics
        
    def validate_epoch(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dict of validation metrics
        """
        self.eval()
        val_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'num_samples': 0,
            'predictions': [],
            'targets': []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Validation step
                metrics = self.validation_step(batch)
                
                # Update validation metrics
                batch_size = batch[0].size(0)
                val_metrics['loss'] += metrics['loss'].item() * batch_size
                val_metrics['accuracy'] += metrics['accuracy'].item() * batch_size
                val_metrics['num_samples'] += batch_size
                
                # Store predictions and targets
                val_metrics['predictions'].extend(metrics['predictions'].cpu().numpy())
                val_metrics['targets'].extend(batch[1].cpu().numpy())
        
        # Calculate averages
        for key in ['loss', 'accuracy']:
            val_metrics[key] /= val_metrics['num_samples']
            
        return val_metrics
        
    def save_checkpoint(self, path: str, optimizer: torch.optim.Optimizer, 
                       epoch: int, metrics: Dict[str, Any]):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            optimizer: Model optimizer
            epoch: Current epoch
            metrics: Current metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint to: {path}")
        
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Dict containing checkpoint data
        """
        logging.info(f"Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
        
    @abstractmethod
    def get_model_specific_args(self) -> Dict[str, Any]:
        """Get model-specific arguments.
        
        Returns:
            Dict of argument names and default values
        """
        pass