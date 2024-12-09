import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from typing import Dict, Any, Tuple, Optional

def get_optimizer(model: torch.nn.Module, 
                 optimizer_name: str = 'adam',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5) -> torch.optim.Optimizer:
    """Get optimizer with weight decay handled properly."""
    # Separate parameters with and without weight decay
    decay = set()
    no_decay = set()
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn  # full param name
            
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight'):
                if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                    decay.add(fpn)
                elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                    no_decay.add(fpn)
                    
    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)],
         "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)],
         "weight_decay": 0.0},
    ]
    
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(optim_groups, lr=learning_rate)
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(optim_groups, lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer: Optimizer,
                 scheduler_name: str = 'onecycle',
                 num_epochs: int = 50,
                 steps_per_epoch: int = None,
                 min_lr: float = 1e-6) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler."""
    if scheduler_name == 'onecycle':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        return OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,  # Warm up for 30% of training
            div_factor=25,  # Initial lr = max_lr/25
            final_div_factor=1e4,  # Min lr = initial_lr/10000
        )
    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            min_lr=min_lr,
            verbose=True
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

class EarlyStopping:
    """Early stopping handler."""
    def __init__(self, patience: int = 7, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

class LabelSmoothingLoss(torch.nn.Module):
    """Label smoothing loss."""
    def __init__(self, classes: int, smoothing: float = 0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1)) 