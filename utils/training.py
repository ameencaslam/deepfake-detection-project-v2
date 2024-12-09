import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import logging
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
import time
from contextlib import contextmanager
import warnings
from torch.utils.data import DataLoader
import math
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

@dataclass
class EarlyStoppingState:
    """State for early stopping."""
    patience: int
    min_delta: float = 0.0
    mode: str = 'min'
    best_value: float = float('inf')
    counter: int = 0
    best_epoch: int = 0
    stopped_epoch: int = 0
    should_stop: bool = False

@dataclass
class TrainingState:
    """Training state tracking."""
    epoch: int = 0
    step: int = 0
    best_metric: float = float('inf')
    early_stopping: Optional[EarlyStoppingState] = None
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    loss_scale: float = 0.0
    metrics: Dict[str, List[float]] = field(default_factory=lambda: {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'learning_rates': [], 'gradient_norms': []
    })

class GradientAccumulator:
    """Handles gradient accumulation for larger effective batch sizes."""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.loss_sum = 0.0
        self.should_update = False
        
    def step(self, loss: torch.Tensor) -> float:
        """Process one accumulation step."""
        self.current_step += 1
        self.loss_sum += loss.item()
        
        if self.current_step >= self.accumulation_steps:
            self.should_update = True
            avg_loss = self.loss_sum / self.accumulation_steps
            self.reset()
            return avg_loss
            
        self.should_update = False
        return 0.0
        
    def reset(self):
        """Reset accumulator state."""
        self.current_step = 0
        self.loss_sum = 0.0
        self.should_update = False

class LRFinder:
    """Learning rate finder using the 1cycle policy."""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Initialize logging
        self.history = {'lr': [], 'loss': []}
        self.best_loss = float('inf')
        self.memory = {'best_params': None, 'best_opt_state': None}
        
    def range_test(self,
                  train_loader: DataLoader,
                  start_lr: float = 1e-7,
                  end_lr: float = 10,
                  num_iter: int = 100,
                  step_mode: str = 'exp',
                  smooth_f: float = 0.05,
                  diverge_th: float = 5.0):
        """Performs the learning rate range test."""
        # Save model state
        self.memory['best_params'] = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.memory['best_opt_state'] = self.optimizer.state_dict()
        
        # Reset optimizer and initialize lr
        self.optimizer.state = {}
        self.optimizer.param_groups[0]['lr'] = start_lr
        
        # Calculate lr schedule
        if step_mode == 'exp':
            lr_schedule = np.exp(np.linspace(np.log(start_lr), np.log(end_lr), num_iter))
        else:
            lr_schedule = np.linspace(start_lr, end_lr, num_iter)
            
        # Training loop
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= num_iter:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Log lr and loss
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            loss_value = loss.item()
            
            # Smooth loss value
            if batch_idx == 0:
                self.history['loss'].append(loss_value)
            else:
                loss_value = smooth_f * loss_value + (1 - smooth_f) * self.history['loss'][-1]
                self.history['loss'].append(loss_value)
                
            # Check if loss has diverged
            if loss_value > diverge_th * self.best_loss:
                break
                
            if loss_value < self.best_loss:
                self.best_loss = loss_value
                
            self.optimizer.step()
            
            # Update lr
            if batch_idx < len(lr_schedule):
                self.optimizer.param_groups[0]['lr'] = lr_schedule[batch_idx]
                
        # Restore model and optimizer state
        self.model.load_state_dict(self.memory['best_params'])
        self.optimizer.load_state_dict(self.memory['best_opt_state'])
        
    def plot(self, skip_start: int = 10, skip_end: int = 5):
        """Plots the learning rate range test results."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['lr'][skip_start:-skip_end],
                self.history['loss'][skip_start:-skip_end])
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Range Test')
        plt.grid(True)
        return plt
        
    def suggest_lr(self, skip_start: int = 10, skip_end: int = 5) -> float:
        """Suggests the optimal learning rate."""
        losses = np.array(self.history['loss'])
        lrs = np.array(self.history['lr'])
        
        # Trim the beginning and end of the data
        losses = losses[skip_start:-skip_end]
        lrs = lrs[skip_start:-skip_end]
        
        # Find the point of steepest descent
        gradients = np.gradient(losses)
        steepest_idx = np.argmin(gradients)
        
        return lrs[steepest_idx]

class TrainingController:
    """Controls and manages the training process."""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 config: Dict[str, Any]):
        """Initialize training controller.
        
        Args:
            model: Model to train
            optimizer: Optimizer instance
            criterion: Loss criterion
            device: Device to train on
            config: Training configuration
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        
        # Initialize training state
        self.state = TrainingState()
        
        # Setup early stopping if enabled
        if config.get('early_stopping_patience', 0) > 0:
            self.state.early_stopping = EarlyStoppingState(
                patience=config['early_stopping_patience'],
                min_delta=config.get('early_stopping_min_delta', 0.0),
                mode=config.get('early_stopping_mode', 'min')
            )
            
        # Setup gradient accumulation
        self.grad_accumulator = GradientAccumulator(
            config.get('gradient_accumulation_steps', 1)
        )
        
        # Setup mixed precision training
        self.use_mixed_precision = config.get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Setup gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', None)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f'Epoch {self.state.epoch}') as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Mixed precision context
                with autocast() if self.use_mixed_precision else contextmanager(lambda: (yield))():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                # Scale loss for gradient accumulation
                loss = loss / self.config.get('gradient_accumulation_steps', 1)
                
                # Backward pass with mixed precision if enabled
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                # Gradient accumulation
                avg_loss = self.grad_accumulator.step(loss)
                
                if self.grad_accumulator.should_update:
                    if self.max_grad_norm is not None:
                        if self.use_mixed_precision:
                            self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.state.gradient_norm = grad_norm.item()
                        
                    # Optimizer step with mixed precision if enabled
                    if self.use_mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    self.optimizer.zero_grad()
                    
                    # Update metrics
                    total_loss += avg_loss
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': avg_loss,
                        'acc': 100. * correct / total,
                        'lr': self.optimizer.param_groups[0]['lr']
                    })
                    
                self.state.step += 1
                
        # Calculate epoch metrics
        metrics = {
            'train_loss': total_loss / len(train_loader),
            'train_acc': 100. * correct / total,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'gradient_norm': self.state.gradient_norm
        }
        
        # Update training state
        for key, value in metrics.items():
            if key in self.state.metrics:
                self.state.metrics[key].append(value)
                
        return metrics
        
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Mixed precision context
            with autocast() if self.use_mixed_precision else contextmanager(lambda: (yield))():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        # Calculate validation metrics
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_acc': 100. * correct / total
        }
        
        # Update training state
        for key, value in metrics.items():
            if key in self.state.metrics:
                self.state.metrics[key].append(value)
                
        # Check early stopping
        if self.state.early_stopping is not None:
            self._check_early_stopping(metrics['val_loss'])
            
        return metrics
        
    def _check_early_stopping(self, current_value: float) -> bool:
        """Check if training should stop early."""
        es = self.state.early_stopping
        
        if es.mode == 'min':
            improved = current_value < es.best_value - es.min_delta
        else:
            improved = current_value > es.best_value + es.min_delta
            
        if improved:
            es.best_value = current_value
            es.counter = 0
            es.best_epoch = self.state.epoch
        else:
            es.counter += 1
            
        if es.counter >= es.patience:
            es.should_stop = True
            es.stopped_epoch = self.state.epoch
            self.logger.info(
                f'Early stopping triggered. Best value: {es.best_value:.6f} '
                f'at epoch {es.best_epoch}'
            )
            
        return es.should_stop
        
    def save_state(self, path: Union[str, Path]):
        """Save training state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state_dict = {
            'epoch': self.state.epoch,
            'step': self.state.step,
            'best_metric': self.state.best_metric,
            'metrics': self.state.metrics,
            'early_stopping': self.state.early_stopping.__dict__ if self.state.early_stopping else None,
            'gradient_norm': self.state.gradient_norm,
            'learning_rate': self.state.learning_rate,
            'loss_scale': self.state.loss_scale
        }
        
        torch.save(state_dict, path)
        
    def load_state(self, path: Union[str, Path]):
        """Load training state."""
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Training state file not found: {path}")
            
        state_dict = torch.load(path)
        
        self.state.epoch = state_dict['epoch']
        self.state.step = state_dict['step']
        self.state.best_metric = state_dict['best_metric']
        self.state.metrics = state_dict['metrics']
        self.state.gradient_norm = state_dict['gradient_norm']
        self.state.learning_rate = state_dict['learning_rate']
        self.state.loss_scale = state_dict['loss_scale']
        
        if state_dict['early_stopping'] is not None:
            if self.state.early_stopping is None:
                self.state.early_stopping = EarlyStoppingState(**state_dict['early_stopping'])
            else:
                self.state.early_stopping.__dict__.update(state_dict['early_stopping'])
                
    def get_lr_scheduler(self, num_epochs: int, steps_per_epoch: int) -> OneCycleLR:
        """Create learning rate scheduler."""
        return OneCycleLR(
            self.optimizer,
            max_lr=self.config.get('max_lr', 0.1),
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.config.get('warmup_pct', 0.3),
            anneal_strategy='cos',
            div_factor=self.config.get('div_factor', 25.0),
            final_div_factor=self.config.get('final_div_factor', 1e4)
        )
        
    def find_lr(self, train_loader: DataLoader) -> float:
        """Find optimal learning rate."""
        lr_finder = LRFinder(
            self.model, self.optimizer, self.criterion, self.device
        )
        
        lr_finder.range_test(
            train_loader,
            start_lr=self.config.get('lr_finder_start', 1e-7),
            end_lr=self.config.get('lr_finder_end', 10),
            num_iter=self.config.get('lr_finder_iter', 100)
        )
        
        suggested_lr = lr_finder.suggest_lr()
        self.logger.info(f"Suggested learning rate: {suggested_lr:.2e}")
        
        # Save plot if directory specified
        if 'lr_finder_plot_dir' in self.config:
            plot = lr_finder.plot()
            plot.savefig(Path(self.config['lr_finder_plot_dir']) / 'lr_finder.png')
            plt.close()
            
        return suggested_lr