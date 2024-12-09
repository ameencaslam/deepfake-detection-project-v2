from typing import Dict, Any, Optional, List, Union
import time
from dataclasses import dataclass
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from .hardware import HardwareManager
from IPython.display import display, clear_output
import ipywidgets as widgets
import matplotlib.pyplot as plt

@dataclass
class TrainingMetrics:
    """Training metrics container."""
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = float('inf')
    val_acc: float = 0.0
    best_val_loss: float = float('inf')
    best_val_acc: float = 0.0
    best_epoch: int = 0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics from dictionary."""
        for key, value in metrics.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            key: getattr(self, key)
            for key in self.__annotations__
        }

class EarlyStopping:
    """Early stopping handler."""
    def __init__(self,
                 patience: int = 5,
                 min_delta: float = 0.0,
                 mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, value: float, epoch: int) -> bool:
        """Check if training should stop."""
        if self.mode == 'min':
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.counter = 0
                self.best_epoch = epoch
                return False
        else:
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.counter = 0
                self.best_epoch = epoch
                return False
                
        self.counter += 1
        return self.counter >= self.patience

class LRFinder:
    """Learning rate finder using the 1cycle policy approach."""
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Store initial learning rate
        self.init_lr = optimizer.param_groups[0]['lr']
        
        # Results
        self.lr_history = []
        self.loss_history = []
        
    def range_test(self,
                  train_loader: torch.utils.data.DataLoader,
                  start_lr: float = 1e-7,
                  end_lr: float = 10,
                  num_iter: int = 100,
                  step_mode: str = 'exp',
                  smooth_f: float = 0.05,
                  diverge_th: float = 5):
        """
        Perform learning rate range test.
        
        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Maximum learning rate
            num_iter: Number of iterations
            step_mode: 'exp' for exponential increase, 'linear' for linear increase
            smooth_f: Loss smoothing factor
            diverge_th: Threshold for divergence (ratio of maximum loss to minimum loss)
        """
        # Reset model and optimizer
        self.model.train()
        self.optimizer.param_groups[0]['lr'] = start_lr
        
        # Initialize variables
        n_iter = 0
        min_loss = float('inf')
        best_lr = None
        running_loss = None
        
        with tqdm(total=num_iter, desc='Finding learning rate') as pbar:
            for inputs, targets in train_loader:
                if n_iter >= num_iter:
                    break
                    
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_history.append(current_lr)
                
                # Forward pass
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Smooth loss
                if running_loss is None:
                    running_loss = loss.item()
                else:
                    running_loss = running_loss * (1 - smooth_f) + loss.item() * smooth_f
                self.loss_history.append(running_loss)
                
                # Check for divergence
                if running_loss > diverge_th * min_loss:
                    break
                    
                if running_loss < min_loss:
                    min_loss = running_loss
                    best_lr = current_lr
                    
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update learning rate
                if step_mode == 'exp':
                    self.optimizer.param_groups[0]['lr'] = start_lr * (end_lr/start_lr) ** (n_iter/num_iter)
                else:
                    self.optimizer.param_groups[0]['lr'] = start_lr + (end_lr-start_lr) * (n_iter/num_iter)
                    
                n_iter += 1
                pbar.update(1)
                
        # Reset learning rate
        self.optimizer.param_groups[0]['lr'] = self.init_lr
        
        return best_lr
        
    def plot(self, skip_start: int = 10, skip_end: int = 5):
        """Plot learning rate vs loss."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.lr_history[skip_start:-skip_end],
                self.loss_history[skip_start:-skip_end])
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)
        plt.show()

class ProgressTracker:
    def __init__(self,
                 num_epochs: int,
                 num_batches: int,
                 hardware_manager: Optional[HardwareManager] = None,
                 early_stopping_patience: int = 5,
                 save_dir: Optional[Union[str, Path]] = None):
        """Initialize progress tracker."""
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.hardware_manager = hardware_manager
        self.save_dir = Path(save_dir) if save_dir else None
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics
        self.metrics = TrainingMetrics()
        self.history = []
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='min'
        )
        
        # Initialize timing
        self.epoch_start_time = None
        self.training_start_time = time.time()
        
        # Create save directory
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
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
            leave=True
        )
        
    def reset_metrics(self):
        """Reset running metrics for new epoch."""
        self.running_loss = 0.0
        self.running_acc = 0.0
        self.running_grad_norm = 0.0
        self.count = 0
        
    def update_batch(self,
                    batch_idx: int,
                    loss: float,
                    accuracy: float,
                    pbar: tqdm,
                    grad_norm: Optional[float] = None,
                    learning_rate: Optional[float] = None):
        """Update metrics for current batch."""
        # Update running averages
        self.running_loss = (self.running_loss * self.count + loss) / (self.count + 1)
        self.running_acc = (self.running_acc * self.count + accuracy) / (self.count + 1)
        if grad_norm is not None:
            self.running_grad_norm = (self.running_grad_norm * self.count + grad_norm) / (self.count + 1)
        self.count += 1
        
        # Update progress bar
        postfix = {
            'loss': f"{self.running_loss:.4f}",
            'acc': f"{self.running_acc:.2%}"
        }
        if grad_norm is not None:
            postfix['grad'] = f"{self.running_grad_norm:.2e}"
        if learning_rate is not None:
            postfix['lr'] = f"{learning_rate:.2e}"
            
        pbar.set_postfix(postfix)
        pbar.update(1)
        
        # Log hardware stats if available
        if self.hardware_manager and batch_idx % 10 == 0:
            self.hardware_manager.get_hardware_stats()
            
    def end_epoch(self, val_metrics: Dict[str, float]) -> bool:
        """End epoch and display summary. Returns True if training should stop."""
        # Update metrics
        self.metrics.train_loss = self.running_loss
        self.metrics.train_acc = self.running_acc
        self.metrics.val_loss = val_metrics['loss']
        self.metrics.val_acc = val_metrics['accuracy']
        
        # Check for best model
        if val_metrics['loss'] < self.metrics.best_val_loss:
            self.metrics.best_val_loss = val_metrics['loss']
            self.metrics.best_val_acc = val_metrics['accuracy']
            self.metrics.best_epoch = self.current_epoch
            
        # Save history
        self.history.append(self.metrics.to_dict())
        
        # Calculate timing
        epoch_time = time.time() - self.epoch_start_time
        elapsed_time = time.time() - self.training_start_time
        remaining_epochs = self.num_epochs - (self.current_epoch + 1)
        estimated_remaining = (elapsed_time / (self.current_epoch + 1)) * remaining_epochs
        
        # Display epoch summary
        self._display_epoch_summary(epoch_time, estimated_remaining)
        
        # Save progress if directory is set
        if self.save_dir:
            self.save_progress()
            
        # Check early stopping
        should_stop = self.early_stopping(val_metrics['loss'], self.current_epoch)
        if should_stop:
            self.logger.info(
                f"Early stopping triggered. Best validation loss: {self.metrics.best_val_loss:.4f} "
                f"at epoch {self.metrics.best_epoch + 1}"
            )
            
        return should_stop
        
    def _display_epoch_summary(self, epoch_time: float, estimated_remaining: float):
        """Display epoch summary."""
        print("\nEpoch Summary:")
        print("├── Training Metrics:")
        print(f"│   ├── Loss: {self.metrics.train_loss:.4f}")
        print(f"│   └── Accuracy: {self.metrics.train_acc:.2%}")
        print("├── Validation Metrics:")
        print(f"│   ├── Loss: {self.metrics.val_loss:.4f}")
        print(f"│   └── Accuracy: {self.metrics.val_acc:.2%}")
        print("├── Best Model:")
        print(f"│   ├── Epoch: {self.metrics.best_epoch + 1}")
        print(f"│   ├── Val Loss: {self.metrics.best_val_loss:.4f}")
        print(f"│   └── Val Accuracy: {self.metrics.best_val_acc:.2%}")
        print("└── Time Statistics:")
        print(f"    ├── Epoch Time: {epoch_time:.1f}s")
        print(f"    └── Estimated Remaining: {estimated_remaining/3600:.1f}h")
        
        if self.hardware_manager:
            print("\nHardware Statistics:")
            self.hardware_manager.print_hardware_info()
            
    def save_progress(self):
        """Save training progress to file."""
        progress_file = self.save_dir / 'training_progress.json'
        progress_data = {
            'current_epoch': self.current_epoch,
            'best_epoch': self.metrics.best_epoch,
            'best_val_loss': self.metrics.best_val_loss,
            'best_val_acc': self.metrics.best_val_acc,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=4)
            
    def plot_progress(self):
        """Plot training progress."""
        if not self.history:
            return
            
        epochs = range(1, len(self.history) + 1)
        metrics = ['loss', 'acc']
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
            
        for ax, metric in zip(axes, metrics):
            ax.plot(epochs, [h[f'train_{metric}'] for h in self.history], 'b-',
                   label=f'Training {metric}')
            ax.plot(epochs, [h[f'val_{metric}'] for h in self.history], 'r-',
                   label=f'Validation {metric}')
            ax.set_title(f'Training and Validation {metric.capitalize()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True)
            
        plt.tight_layout()
        if self.save_dir:
            plt.savefig(self.save_dir / 'training_progress.png')
        plt.show()

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
        self.status = widgets.HTML(value='Training in progress...')
        
        # Create container for button and status
        self.container = widgets.VBox([
            self.button,
            self.status
        ])
        display(self.container)
        
    def _stop_clicked(self, _):
        """Handle stop button click."""
        self._should_stop = True
        self.button.description = 'Stopping...'
        self.button.disabled = True
        self.status.value = '<span style="color: orange;">Stopping after current epoch...</span>'
        
    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self._should_stop
        
    def cleanup(self):
        """Clean up widgets."""
        clear_output()
        
    def update_status(self, message: str, color: str = 'black'):
        """Update status message."""
        self.status.value = f'<span style="color: {color};">{message}</span>'