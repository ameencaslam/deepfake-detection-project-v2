import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import timm
from ..base_model import BaseModel

class AttentionBlock(nn.Module):
    """Squeeze-and-Excitation attention block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EfficientNetModel(BaseModel):
    """EfficientNet model with custom enhancements."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Load base model
        self.base_model = timm.create_model(
            'tf_efficientnet_b3_ns',
            pretrained=config['model']['pretrained'],
            num_classes=0,  # Remove classifier
            drop_rate=config['model']['dropout_rate'],
            drop_path_rate=0.2
        )
        
        # Get feature dimensions
        self.features_dim = self.base_model.num_features
        
        # Add attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(self.features_dim // 2),
            AttentionBlock(self.features_dim)
        ])
        
        # Add classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=config['model']['dropout_rate']),
            nn.Linear(self.features_dim, config['model']['num_classes'])
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Enable gradient checkpointing if configured
        if config['model'].get('gradient_checkpointing', False):
            self.enable_gradient_checkpointing()
            
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features."""
        features = []
        
        # Get base model features
        for i, block in enumerate(self.base_model.blocks):
            x = block(x)
            if i in [3, 5]:  # Extract features from middle and final blocks
                features.append(x)
                
        # Apply attention
        for i, (feature, attention) in enumerate(zip(features, self.attention_blocks)):
            features[i] = attention(feature)
            
        return features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention and feature extraction."""
        # Extract base features
        features = self.base_model.forward_features(x)
        
        # Apply final attention
        features = self.attention_blocks[-1](features)
        
        # Classification
        out = self.classifier(features)
        
        return out
        
    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get attention weights for visualization."""
        attention_weights = {}
        
        # Get base features
        features = self.base_model.forward_features(x)
        
        # Get attention weights from each block
        for i, attention in enumerate(self.attention_blocks):
            # Get attention scores
            b, c, _, _ = features.size()
            scores = attention.avg_pool(features).view(b, c)
            scores = attention.fc(scores).view(b, c, 1, 1)
            
            attention_weights[f'block_{i}'] = scores.squeeze(-1).squeeze(-1)
            
        return attention_weights
        
    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone layers."""
        for param in self.base_model.parameters():
            param.requires_grad = not freeze
            
    def unfreeze_stages(self, stages: List[int]):
        """Unfreeze specific stages of the backbone."""
        if not isinstance(stages, (list, tuple)):
            stages = [stages]
            
        for i, block in enumerate(self.base_model.blocks):
            if i in stages:
                for param in block.parameters():
                    param.requires_grad = True
                    
    def get_layer_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get outputs from different layers for visualization."""
        outputs = {}
        
        # Get outputs from each block
        for i, block in enumerate(self.base_model.blocks):
            x = block(x)
            outputs[f'block_{i}'] = x
            
        # Get attention outputs
        features = self.base_model.forward_features(x)
        for i, attention in enumerate(self.attention_blocks):
            outputs[f'attention_{i}'] = attention(features)
            
        return outputs
        
    def configure_optimizers(self, lr: float = 1e-4, weight_decay: float = 0.01) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizer with layer-wise learning rates."""
        # Create parameter groups with different learning rates
        backbone_params = []
        attention_params = []
        classifier_params = []
        
        # Backbone parameters (lower learning rate)
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
                
        # Attention parameters (medium learning rate)
        for block in self.attention_blocks:
            attention_params.extend(block.parameters())
            
        # Classifier parameters (higher learning rate)
        classifier_params.extend(self.classifier.parameters())
        
        # Create parameter groups
        param_groups = [
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': attention_params, 'lr': lr * 0.5},
            {'params': classifier_params, 'lr': lr}
        ]
        
        # Create optimizer
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[lr * 0.1, lr * 0.5, lr],
            epochs=self.config['training']['num_epochs'],
            steps_per_epoch=self.config['training']['steps_per_epoch'],
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        return optimizer, scheduler
        
    def get_complexity_info(self) -> Dict[str, int]:
        """Get model complexity information."""
        from ptflops import get_model_complexity_info
        
        macs, params = get_model_complexity_info(
            self,
            (3, 224, 224),
            as_strings=False,
            print_per_layer_stat=False
        )
        
        return {
            'macs': macs,
            'params': params,
            'backbone_params': sum(p.numel() for p in self.base_model.parameters()),
            'attention_params': sum(p.numel() for p in self.attention_blocks.parameters()),
            'classifier_params': sum(p.numel() for p in self.classifier.parameters())
        }
        
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature importance scores using integrated gradients."""
        self.eval()
        x.requires_grad = True
        
        # Get baseline (zero input)
        baseline = torch.zeros_like(x)
        
        # Number of steps for approximation
        steps = 50
        
        # Generate scaled inputs
        scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(steps + 1)]
        
        # Get gradients for each scaled input
        grads = []
        for scaled_input in scaled_inputs:
            output = self(scaled_input)
            grad = torch.autograd.grad(output.sum(), scaled_input)[0]
            grads.append(grad)
            
        # Average gradients
        avg_grads = torch.stack(grads).mean(0)
        
        # Calculate feature importance
        importance = (x - baseline) * avg_grads
        
        return importance.abs().mean(dim=1)  # Average over channels
        
    def __str__(self) -> str:
        """String representation of model."""
        complexity = self.get_complexity_info()
        description = [
            "EfficientNet Model:",
            f"Architecture: {self.config['model']['architecture']}",
            f"Parameters: {complexity['params']:,}",
            f"MACs: {complexity['macs']:,}",
            f"Backbone Parameters: {complexity['backbone_params']:,}",
            f"Attention Parameters: {complexity['attention_params']:,}",
            f"Classifier Parameters: {complexity['classifier_params']:,}"
        ]
        return "\n".join(description)