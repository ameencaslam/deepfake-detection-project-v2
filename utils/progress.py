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
    def __init__(self, num_epochs: int, num_batches: int, hardware_manager: HardwareManager):
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.hardware_manager = hardware_manager
        self.metrics = TrainingMetrics()
        self.epoch_times = []
        
    def new_epoch(self, epoch: int) -> tqdm:
        """Initialize a new epoch progress bar."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        
        # Create epoch progress bar
        return tqdm(total=self.num_batches,
                   desc=f"Epoch [{epoch+1}/{self.num_epochs}]",
                   bar_format='{desc}: {percentage:3.0f}%|{bar:30}{r_bar}')
                   
    def update_batch(self, batch_idx: int, loss: float, acc: float, pbar: tqdm):
        """Update batch progress."""
        pbar.update(1)
        self.metrics.train_loss = loss
        self.metrics.train_acc = acc
        
        # Update progress bar description with metrics
        pbar.set_postfix({
            'loss': f"{loss:.4f}",
            'acc': f"{acc*100:.2f}%"
        })
        
    def end_epoch(self, val_metrics: Dict[str, float]):
        """Process end of epoch and display metrics."""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # Update validation metrics
        self.metrics.val_loss = val_metrics['loss']
        self.metrics.val_acc = val_metrics['accuracy']
        
        # Update best metrics
        if self.metrics.val_loss < self.metrics.best_val_loss:
            self.metrics.best_val_loss = self.metrics.val_loss
            self.metrics.best_val_acc = self.metrics.val_acc
            self.metrics.best_epoch = self.current_epoch
            
        self._display_epoch_summary()
        
    def _display_epoch_summary(self):
        """Display detailed epoch summary."""
        # Get hardware stats
        hw_stats = self.hardware_manager.get_hardware_stats()
        
        # Calculate time estimates
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.num_epochs - (self.current_epoch + 1)
        estimated_time = avg_epoch_time * remaining_epochs
        
        print("\nEpoch Summary:")
        print("├── Training Metrics:")
        print(f"│   ├── Loss: {self.metrics.train_loss:.4f}")
        print(f"│   └── Accuracy: {self.metrics.train_acc*100:.2f}%")
        print("├── Validation Metrics:")
        print(f"│   ├── Loss: {self.metrics.val_loss:.4f}")
        print(f"│   └── Accuracy: {self.metrics.val_acc*100:.2f}%")
        print("├── Best Model:")
        print(f"│   ├── Epoch: {self.metrics.best_epoch + 1}")
        print(f"│   ├── Val Loss: {self.metrics.best_val_loss:.4f}")
        print(f"│   └── Val Accuracy: {self.metrics.best_val_acc*100:.2f}%")
        
        if hw_stats.gpu_utilization is not None:
            print("├── Hardware Usage:")
            print(f"│   ├── GPU: {hw_stats.gpu_utilization:.1f}%")
            print(f"│   ├── GPU Memory: {hw_stats.gpu_memory_used:.1f}GB/{hw_stats.gpu_memory_total:.1f}GB")
            print(f"│   └── RAM: {hw_stats.ram_used:.1f}GB/{hw_stats.ram_total:.1f}GB")
            
        print("└── Time Statistics:")
        print(f"    ├── Epoch Time: {avg_epoch_time:.1f}s")
        print(f"    └── Estimated Remaining: {estimated_time/3600:.1f}h")
        print()
        
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current training metrics."""
        return {
            'train_loss': self.metrics.train_loss,
            'train_acc': self.metrics.train_acc,
            'val_loss': self.metrics.val_loss,
            'val_acc': self.metrics.val_acc,
            'best_val_loss': self.metrics.best_val_loss,
            'best_val_acc': self.metrics.best_val_acc,
        }

class TrainingController:
    def __init__(self):
        self.stop_next_epoch = False
        self._create_stop_button()
        
    def _create_stop_button(self):
        """Create and display stop button widget."""
        self.stop_button = widgets.Button(
            description='Stop Training',
            button_style='danger',
            layout=widgets.Layout(width='150px', height='40px')
        )
        self.stop_button.on_click(lambda b: self.request_stop())
        display(self.stop_button)
        
    def request_stop(self):
        """Request training to stop after current epoch."""
        self.stop_next_epoch = True
        self.stop_button.description = 'Stopping...'
        self.stop_button.disabled = True
        print("\nTraining will stop after current epoch...")
        
    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self.stop_next_epoch 