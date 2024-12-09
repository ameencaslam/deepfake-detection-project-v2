from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import os
from torch.cuda.amp import autocast, GradScaler

class BaseModel(nn.Module, ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.mixed_precision = config.get('mixed_precision', True)
        self.scaler = GradScaler() if self.mixed_precision else None
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass
        
    @abstractmethod
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizer and learning rate scheduler."""
        pass
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single training step."""
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Mixed precision training
        with autocast(enabled=self.mixed_precision):
            outputs = self(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': predicted,
        }
        
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single validation step."""
        self.eval()
        with torch.no_grad():
            metrics = self.training_step(batch)
        self.train()
        return metrics
        
    def save_checkpoint(self, path: str, epoch: int, optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler,
                       metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint
        
    def get_progress_bar_dict(self) -> Dict[str, Any]:
        """Get metrics for progress bar display."""
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model specific arguments to parser."""
        parser = parent_parser.add_argument_group("BaseModel")
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        return parent_parser 