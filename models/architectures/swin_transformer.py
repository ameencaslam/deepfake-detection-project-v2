import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import timm
from ..base_model import BaseModel

class CrossAttention(nn.Module):
    """Cross-attention module for feature fusion."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        H = self.num_heads
        
        # Project queries, keys, and values
        q = self.q(x).reshape(B, N, H, C // H).permute(0, 2, 1, 3)
        k = self.k(context).reshape(B, N, H, C // H).permute(0, 2, 1, 3)
        v = self.v(context).reshape(B, N, H, C // H).permute(0, 2, 1, 3)
        
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x, attn

class FeatureFusionBlock(nn.Module):
    """Feature fusion block combining local and global features."""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Cross attention
        residual = x
        x = self.norm1(x)
        context = self.norm1(context)
        x, _ = self.cross_attn(x, context)
        x = x + residual
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        
        return x

class SwinTransformerModel(BaseModel):
    """Enhanced Swin Transformer with hierarchical feature fusion."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Load base model
        self.base_model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=config['model']['pretrained'],
            num_classes=0,
            drop_rate=config['model']['dropout_rate'],
            drop_path_rate=0.2
        )
        
        # Get feature dimensions
        self.features_dim = self.base_model.num_features
        
        # Feature fusion blocks for each stage
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(dim=96),   # Stage 1
            FeatureFusionBlock(dim=192),  # Stage 2
            FeatureFusionBlock(dim=384),  # Stage 3
            FeatureFusionBlock(dim=768)   # Stage 4
        ])
        
        # Global context pool
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.features_dim),
            nn.Linear(self.features_dim, config['model']['num_classes'])
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Enable gradient checkpointing if configured
        if config['model'].get('gradient_checkpointing', False):
            self.enable_gradient_checkpointing()
            
    def _initialize_weights(self):
        """Initialize model weights."""
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
        self.apply(_init_weights)
        
    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract hierarchical features."""
        features = []
        
        # Get features from each stage
        x = self.base_model.patch_embed(x)
        x = self.base_model.pos_drop(x)
        
        for i, stage in enumerate(self.base_model.layers):
            x = stage(x)
            features.append(x)
            
        return features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with hierarchical feature fusion."""
        # Extract features from each stage
        features = self.get_features(x)
        
        # Apply feature fusion at each stage
        fused_features = []
        for i, feature in enumerate(features):
            # Get global context
            global_context = self.global_pool(feature.transpose(1, 2)).transpose(1, 2)
            
            # Apply fusion
            fused = self.fusion_blocks[i](feature, global_context)
            fused_features.append(fused)
            
        # Combine features
        x = fused_features[-1]  # Use final stage features
        
        # Global pooling
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        
        # Classification
        x = self.classifier(x)
        
        return x
        
    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get attention maps from all stages."""
        attention_maps = {}
        
        # Get features and attention maps from each stage
        features = self.get_features(x)
        
        for i, feature in enumerate(features):
            # Get global context
            global_context = self.global_pool(feature.transpose(1, 2)).transpose(1, 2)
            
            # Get attention maps from fusion block
            _, attn = self.fusion_blocks[i].cross_attn(feature, global_context)
            attention_maps[f'stage_{i+1}'] = attn
            
        return attention_maps
        
    def configure_optimizers(self, lr: float = 1e-4, weight_decay: float = 0.01) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizer with layer-wise learning rates."""
        # Create parameter groups with different learning rates
        backbone_params = []
        fusion_params = []
        classifier_params = []
        
        # Backbone parameters (lower learning rate)
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
                
        # Fusion parameters (medium learning rate)
        for block in self.fusion_blocks:
            fusion_params.extend(block.parameters())
            
        # Classifier parameters (higher learning rate)
        classifier_params.extend(self.classifier.parameters())
        
        # Create parameter groups
        param_groups = [
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': fusion_params, 'lr': lr * 0.5},
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
        
    def freeze_stages(self, num_stages: int):
        """Freeze first n stages of the backbone."""
        if num_stages <= 0:
            return
            
        # Freeze patch embedding
        if num_stages >= 1:
            for param in self.base_model.patch_embed.parameters():
                param.requires_grad = False
                
        # Freeze stages
        for i in range(min(num_stages - 1, len(self.base_model.layers))):
            for param in self.base_model.layers[i].parameters():
                param.requires_grad = False
                
    def get_intermediate_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate features for visualization."""
        features = {}
        
        # Patch embedding
        x = self.base_model.patch_embed(x)
        features['patch_embed'] = x
        
        # Position embedding
        x = self.base_model.pos_drop(x)
        
        # Stages
        for i, stage in enumerate(self.base_model.layers):
            x = stage(x)
            features[f'stage_{i+1}'] = x
            
            # Get fused features
            global_context = self.global_pool(x.transpose(1, 2)).transpose(1, 2)
            fused = self.fusion_blocks[i](x, global_context)
            features[f'fused_stage_{i+1}'] = fused
            
        return features
        
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
            'fusion_params': sum(p.numel() for p in self.fusion_blocks.parameters()),
            'classifier_params': sum(p.numel() for p in self.classifier.parameters())
        }
        
    def __str__(self) -> str:
        """String representation of model."""
        complexity = self.get_complexity_info()
        description = [
            "Swin Transformer Model:",
            f"Architecture: {self.config['model']['architecture']}",
            f"Parameters: {complexity['params']:,}",
            f"MACs: {complexity['macs']:,}",
            f"Backbone Parameters: {complexity['backbone_params']:,}",
            f"Fusion Parameters: {complexity['fusion_params']:,}",
            f"Classifier Parameters: {complexity['classifier_params']:,}"
        ]
        return "\n".join(description)