from typing import Dict, Any, Optional
import time
from dataclasses import dataclass
from tqdm import tqdm
import torch
from .hardware import HardwareManager
from IPython.display import display
import ipywidgets as widgets

@dataclass
class TrainingMetrics:
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = float('inf')
    val_acc: float = 0.0
    best_val_loss: float = float('inf')
    best_val_acc: float = 0.0
    best_epoch: int = 0

class ProgressTracker:
    def __init__(self, num_epochs: int, num_batches: int, hardware_manager=None):
        """Initialize progress tracker."""
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.hardware_manager = hardware_manager
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.current_epoch = 0
        self.epoch_start_time = None
        self.training_start_time = time.time()
        
        # Initialize metrics
        self.reset_metrics()
        
    def reset_metrics(self):
        """Reset running metrics for new epoch."""
        self.train_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'num_samples': 0
        }
        
    def new_epoch(self, epoch: int):
        """Start tracking new epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.reset_metrics()
        
        # Create progress bar
        return tqdm(
            total=self.num_batches,
            desc=f"Epoch [{epoch+1}/{self.num_epochs}]",
            unit='batch',
            dynamic_ncols=True,
            leave=True  # Keep the progress bar after completion
        )
        
    def update_batch(self, batch_idx: int, metrics: Dict[str, float], pbar):
        """Update metrics for current batch."""
        # Update training metrics
        self.train_metrics['loss'] = metrics['loss']
        self.train_metrics['accuracy'] = metrics['accuracy']
        self.train_metrics['num_samples'] = metrics['num_samples']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'acc': f"{metrics['accuracy']:.2%}"
        })
        pbar.update(1)
        
    def end_epoch(self, val_metrics: Dict[str, float]):
        """End epoch and display summary."""
        epoch_time = time.time() - self.epoch_start_time
        
        # Update best metrics based on validation loss
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_epoch = self.current_epoch
            
        # Calculate time statistics
        elapsed_time = time.time() - self.training_start_time
        remaining_epochs = self.num_epochs - (self.current_epoch + 1)
        estimated_remaining = (elapsed_time / (self.current_epoch + 1)) * remaining_epochs
        
        # Display epoch summary
        print("\nEpoch Summary:")
        print("├── Training Metrics:")
        print(f"│   ├── Loss: {self.train_metrics['loss']:.4f}")
        print(f"│   └── Accuracy: {self.train_metrics['accuracy']:.2%}")
        print("├── Validation Metrics:")
        print(f"│   ├── Loss: {val_loss:.4f}")
        print(f"│   └── Accuracy: {val_acc:.2%}")
        print("├── Best Model (by val loss):")
        print(f"│   ├── Epoch: {self.best_epoch + 1}")
        print(f"│   ├── Val Loss: {self.best_val_loss:.4f}")
        print(f"│   └── Val Accuracy: {self.best_val_acc:.2%}")
        print("└── Time Statistics:")
        print(f"    ├── Epoch Time: {epoch_time:.1f}s")
        print(f"    └── Estimated Remaining: {estimated_remaining/3600:.1f}h")
        print("\n")
        
        return is_best  # Return whether this is the best model

class TrainingController:
    def __init__(self):
        """Initialize training controller with stop button."""
        self._should_stop = False
        self.button = widgets.Button(
            description='Stop Training',
            button_style='danger',
            layout=widgets.Layout(width='150px', height='40px')
        )
        self.button.on_click(self._stop_clicked)
        display(self.button)
        
    def _stop_clicked(self, _):
        """Handle stop button click."""
        self._should_stop = True
        self.button.description = 'Stopping...'
        self.button.disabled = True
        
    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self._should_stop