from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Set
import logging
from pathlib import Path
import json
from torch.cuda.amp import autocast
import gc
import warnings
from dataclasses import dataclass, field
import numpy as np
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import os
import time
from contextlib import contextmanager

@dataclass
class ModelCheckpoint:
    """Checkpoint metadata and state."""
    version: str
    epoch: int
    model_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    scaler_state: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_state: Dict[str, Any] = field(default_factory=dict)

class DimensionValidator:
    """Validates tensor dimensions throughout the model."""
    
    def __init__(self, expected_shapes: Dict[str, Tuple[Optional[int], ...]]):
        """Initialize validator with expected shapes.
        
        Args:
            expected_shapes: Dictionary mapping layer names to expected shapes.
                           Use None for dynamic dimensions.
        """
        self.expected_shapes = expected_shapes
        self.current_shapes = {}
        
    def validate(self, name: str, tensor: torch.Tensor) -> bool:
        """Validate tensor shape against expected shape."""
        if name not in self.expected_shapes:
            return True
            
        expected = self.expected_shapes[name]
        actual = tensor.shape
        
        if len(expected) != len(actual):
            return False
            
        for exp, act in zip(expected, actual):
            if exp is not None and exp != act:
                return False
                
        self.current_shapes[name] = actual
        return True
        
    def get_shape_mismatch(self, name: str, tensor: torch.Tensor) -> str:
        """Get description of shape mismatch."""
        if name not in self.expected_shapes:
            return "No expected shape defined"
            
        expected = self.expected_shapes[name]
        actual = tensor.shape
        
        return f"Expected shape: {expected}, got: {actual}"

class FeatureExtractor:
    """Extract and visualize intermediate features."""
    def __init__(self, model: nn.Module, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.features = {}
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks for feature extraction."""
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
            
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(hook_fn(name)))
                
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from input."""
        self.features = {}
        _ = self.model(x)
        return self.features
        
    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()

class BaseModel(nn.Module, ABC):
    """Enhanced base model with advanced features."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature extraction
        self.feature_extractor = None
        self.visualization_mode = False
        
        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Initialize dimension validation
        self.dimension_validator = None
        
        # Initialize distributed training state
        self.is_distributed = False
        self.local_rank = 0
        
        # Initialize mixed precision training
        self.use_mixed_precision = config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        # Initialize memory optimization
        self.gradient_checkpointing = False
        self.activation_checkpointing = False
        
        # Initialize model quantization
        self.is_quantized = False
        self.qconfig = None
        
    def setup_distributed(self, local_rank: int):
        """Setup distributed training."""
        self.is_distributed = True
        self.local_rank = local_rank
        
        # Convert model to DDP
        self.to(f'cuda:{local_rank}')
        self = DistributedDataParallel(
            self, device_ids=[local_rank],
            output_device=local_rank
        )
        
    def setup_dimension_validation(self, expected_shapes: Dict[str, Tuple[Optional[int], ...]]):
        """Setup dimension validation."""
        self.dimension_validator = DimensionValidator(expected_shapes)
        
    @contextmanager
    def autocast_context(self):
        """Context manager for mixed precision training."""
        if self.use_mixed_precision:
            with autocast():
                yield
        else:
            yield
            
    def validate_dimensions(self, name: str, tensor: torch.Tensor):
        """Validate tensor dimensions."""
        if self.dimension_validator is not None:
            if not self.dimension_validator.validate(name, tensor):
                raise ValueError(
                    f"Shape mismatch at {name}: "
                    f"{self.dimension_validator.get_shape_mismatch(name, tensor)}"
                )
                
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dimension validation."""
        self.validate_dimensions('input', x)
        
    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate features for visualization."""
        pass
        
    def enable_visualization(self, target_layers: Optional[List[str]] = None):
        """Enable feature visualization mode."""
        if target_layers is None:
            # Default layers to visualize
            target_layers = ['features.0', 'features.4', 'features.8']
            
        self.feature_extractor = FeatureExtractor(self, target_layers)
        self.visualization_mode = True
        
    def disable_visualization(self):
        """Disable feature visualization mode."""
        if self.feature_extractor is not None:
            self.feature_extractor.remove_hooks()
            self.feature_extractor = None
        self.visualization_mode = False
        
    def configure_optimizers(self, lr: float, weight_decay: float = 0.01) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizer and learning rate scheduler."""
        # Create parameter groups with different learning rates
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        # Create optimizer
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=self.config.get('num_epochs', 10),
            steps_per_epoch=self.config.get('steps_per_epoch', 100),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        return optimizer, scheduler
        
    def save_checkpoint(self, 
                       path: Union[str, Path],
                       epoch: int,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       scaler: Optional[torch.cuda.amp.GradScaler] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       training_state: Optional[Dict[str, Any]] = None) -> str:
        """Save enhanced model checkpoint.
        
        Args:
            path: Base path for checkpoint
            epoch: Current epoch
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            scaler: Gradient scaler for mixed precision
            metrics: Current metrics
            training_state: Additional training state
            
        Returns:
            Checkpoint version string
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate version
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        version = f"checkpoint_e{epoch}_{timestamp}"
        checkpoint_path = path.parent / f"{path.stem}_{version}.pt"
        
        # Create checkpoint
        checkpoint = ModelCheckpoint(
            version=version,
            epoch=epoch,
            model_state=self.state_dict(),
            optimizer_state=optimizer.state_dict() if optimizer else None,
            scheduler_state=scheduler.state_dict() if scheduler else None,
            scaler_state=scaler.state_dict() if scaler else None,
            metrics=metrics or {},
            hyperparameters=self.config,
            training_state=training_state or {}
        )
        
        # Save checkpoint
        torch.save(checkpoint.__dict__, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return version
        
    def load_checkpoint(self,
                       path: Union[str, Path],
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       scaler: Optional[torch.cuda.amp.GradScaler] = None,
                       strict: bool = True) -> ModelCheckpoint:
        """Load enhanced model checkpoint.
        
        Args:
            path: Path to checkpoint file
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            scaler: Gradient scaler for mixed precision
            strict: Whether to strictly enforce parameter loading
            
        Returns:
            Loaded checkpoint metadata
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Checkpoint not found: {path}")
            
        # Load checkpoint
        checkpoint_dict = torch.load(
            path, map_location=lambda storage, loc: storage
        )
        checkpoint = ModelCheckpoint(**checkpoint_dict)
        
        # Load model state
        missing_keys, unexpected_keys = self.load_state_dict(
            checkpoint.model_state, strict=strict
        )
        
        if missing_keys and strict:
            raise ValueError(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys and strict:
            raise ValueError(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
        # Load optimizer state
        if optimizer is not None and checkpoint.optimizer_state is not None:
            optimizer.load_state_dict(checkpoint.optimizer_state)
            
        # Load scheduler state
        if scheduler is not None and checkpoint.scheduler_state is not None:
            scheduler.load_state_dict(checkpoint.scheduler_state)
            
        # Load scaler state
        if scaler is not None and checkpoint.scaler_state is not None:
            scaler.load_state_dict(checkpoint.scaler_state)
            
        # Load metrics
        self.metrics.update(checkpoint.metrics)
        
        # Update config
        self.config.update(checkpoint.hyperparameters)
        
        self.logger.info(f"Loaded checkpoint: {path}")
        return checkpoint
        
    def get_parameter_count(self) -> Dict[str, int]:
        """Get detailed parameter counts."""
        counts = {
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'frozen': sum(p.numel() for p in self.parameters() if not p.requires_grad)
        }
        
        # Count by layer type
        type_counts = {}
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                type_name = module.__class__.__name__
                params = sum(p.numel() for p in module.parameters())
                type_counts[type_name] = type_counts.get(type_name, 0) + params
                
        counts['by_type'] = type_counts
        return counts
        
    def freeze_layers(self, layers: List[str]):
        """Freeze specified layers."""
        for name, param in self.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = False
                
    def unfreeze_layers(self, layers: List[str]):
        """Unfreeze specified layers."""
        for name, param in self.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = True
                
    @torch.no_grad()
    def get_layer_activations(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Get activations for a specific layer."""
        if self.feature_extractor is None:
            self.enable_visualization([layer_name])
            
        features = self.feature_extractor(x)
        return features.get(layer_name)
        
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        
        # Enable checkpointing for all eligible modules
        for module in self.modules():
            if hasattr(module, 'checkpoint') and callable(module.checkpoint):
                module.checkpoint = True
                
    def enable_activation_checkpointing(self):
        """Enable activation checkpointing for memory efficiency."""
        self.activation_checkpointing = True
        
        def checkpoint_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                return torch.utils.checkpoint.checkpoint(
                    lambda x: x, output
                )
            return output
            
        # Register hooks for activation checkpointing
        for name, module in self.named_modules():
            if any(isinstance(module, t) for t in [nn.ReLU, nn.GELU, nn.Tanh]):
                module.register_forward_hook(checkpoint_hook)
                
    def quantize(self, qconfig: str = 'fbgemm'):
        """Quantize model for inference."""
        if self.is_quantized:
            return
            
        self.qconfig = torch.quantization.get_default_qconfig(qconfig)
        torch.quantization.prepare(self, inplace=True)
        torch.quantization.convert(self, inplace=True)
        self.is_quantized = True
        
    def enable_mixed_precision(self):
        """Enable mixed precision training."""
        self.use_mixed_precision = True
        self.scaler = torch.cuda.amp.GradScaler()
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get model memory statistics."""
        stats = {
            'params_mb': sum(p.numel() * p.element_size() for p in self.parameters()) / 1e6,
            'buffers_mb': sum(b.numel() * b.element_size() for b in self.buffers()) / 1e6
        }
        
        if torch.cuda.is_available():
            stats.update({
                'cuda_allocated_mb': torch.cuda.memory_allocated() / 1e6,
                'cuda_cached_mb': torch.cuda.memory_reserved() / 1e6
            })
            
        return stats
        
    def profile_forward_pass(self, x: torch.Tensor) -> Dict[str, float]:
        """Profile forward pass performance."""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warmup
        for _ in range(3):
            self.forward(x)
            
        # Profile
        timings = []
        memory_peaks = []
        
        for _ in range(10):
            torch.cuda.reset_peak_memory_stats()
            start_event.record()
            
            self.forward(x)
            
            end_event.record()
            torch.cuda.synchronize()
            
            timings.append(start_event.elapsed_time(end_event))
            memory_peaks.append(torch.cuda.max_memory_allocated())
            
        return {
            'mean_time_ms': np.mean(timings),
            'std_time_ms': np.std(timings),
            'mean_memory_mb': np.mean(memory_peaks) / 1e6,
            'std_memory_mb': np.std(memory_peaks) / 1e6
        }
        
    def print_model_summary(self):
        """Print detailed model summary."""
        print("\nModel Summary:")
        print("=" * 50)
        
        # Print architecture
        print("\nArchitecture:")
        print(self)
        
        # Print parameter counts
        counts = self.get_parameter_count()
        print("\nParameter Counts:")
        print(f"Total parameters: {counts['total']:,}")
        print(f"Trainable parameters: {counts['trainable']:,}")
        print(f"Frozen parameters: {counts['frozen']:,}")
        
        print("\nParameters by type:")
        for type_name, count in counts['by_type'].items():
            print(f"{type_name}: {count:,}")
            
        # Print memory stats
        stats = self.get_memory_stats()
        print("\nMemory Statistics:")
        for name, value in stats.items():
            print(f"{name}: {value:.2f} MB")
            
        # Print configuration
        print("\nConfiguration:")
        for key, value in self.config.items():
            print(f"{key}: {value}")
            
    def __str__(self) -> str:
        """Enhanced string representation."""
        lines = [super().__str__()]
        lines.append("\nModel Configuration:")
        lines.extend(f"  {k}: {v}" for k, v in self.config.items())
        
        if self.is_distributed:
            lines.append(f"\nDistributed Training: True (rank {self.local_rank})")
            
        if self.use_mixed_precision:
            lines.append("\nMixed Precision: Enabled")
            
        if self.gradient_checkpointing:
            lines.append("\nGradient Checkpointing: Enabled")
            
        if self.is_quantized:
            lines.append(f"\nQuantized: True (config: {self.qconfig})")
            
        return "\n".join(lines) 