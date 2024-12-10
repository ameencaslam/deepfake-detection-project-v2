import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Tuple, List
from models.base_model import BaseModel
from utils.training import get_optimizer, get_scheduler, LabelSmoothingLoss

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Self attention
        x_norm = self.norm1(x)
        self_attn, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(self_attn)
        
        # Cross attention
        x_norm = self.norm2(x)
        cross_attn, _ = self.cross_attn(x_norm, context, context)
        x = x + self.dropout(cross_attn)
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class CrossAttentionModel(BaseModel):
    def __init__(self,
                 pretrained: bool = True,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3,
                 label_smoothing: float = 0.1):
        config = {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'mixed_precision': True
        }
        super().__init__(config)
        
        # CNN backbone (EfficientNet for multi-scale features)
        self.cnn = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4)  # Get features from multiple scales
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.cnn(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
            self.total_feature_dim = sum(self.feature_dims)
        
        hidden_dim = 256
        
        # Feature processing
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for dim in self.feature_dims
        ])
        
        # Cross attention blocks
        self.cross_attention = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, dropout=dropout_rate)
            for _ in range(2)
        ])
        
        # Global context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.context_proj = nn.Linear(self.total_feature_dim, hidden_dim)
        
        # Feature reduction
        self.feature_reduction = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(512)
        self.feature_dropout = nn.Dropout(dropout_rate)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Label smoothing loss
        self.criterion = LabelSmoothingLoss(num_classes, smoothing=label_smoothing)
        
        # Initialize weights
        self._initialize_weights()
        
        # Save hyperparameters
        self.save_hyperparameters = {
            'architecture': 'cross_attention',
            'pretrained': pretrained,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate,
            'label_smoothing': label_smoothing,
            'feature_dims': self.feature_dims
        }
        
    def _initialize_weights(self):
        """Initialize added layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def _process_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Process multi-scale features with cross attention."""
        # Project features to common dimension
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        
        # Create global context
        global_feats = [self.global_pool(feat).flatten(1) for feat in features]
        global_context = self.context_proj(torch.cat(global_feats, dim=1))
        
        # Process features with cross attention
        processed = []
        for feat in projected:
            # Reshape to sequence
            H, W = feat.shape[-2:]
            feat = feat.flatten(2).transpose(1, 2)  # B, HW, C
            
            # Apply cross attention with global context
            for attn_block in self.cross_attention:
                feat = attn_block(feat, global_context.unsqueeze(1))
                
            processed.append(feat)
            
        # Combine processed features
        combined = torch.cat(processed, dim=1)
        return combined.mean(dim=1)  # Global pooling
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale cross attention."""
        # Get multi-scale features
        features = self.cnn(x)
        
        # Process features with cross attention
        features = self._process_features(features)
        
        # Feature processing
        features = self.feature_reduction(features)
        features = self.feature_norm(features)
        features = self.feature_dropout(features)
        
        # Classification
        return self.classifier(features)
        
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizer with layer-wise learning rates."""
        # Group parameters
        cnn_params = list(self.cnn.parameters())
        attention_params = list(self.cross_attention.parameters()) + \
                         list(self.context_proj.parameters())
        new_params = list(self.projections.parameters()) + \
                    list(self.feature_reduction.parameters()) + \
                    list(self.classifier.parameters()) + \
                    list(self.feature_norm.parameters())
                    
        param_groups = [
            {'params': cnn_params, 'lr': 1e-4},        # Lower LR for pretrained
            {'params': attention_params, 'lr': 3e-4},  # Higher LR for attention
            {'params': new_params, 'lr': 3e-4}         # Higher LR for new layers
        ]
        
        # Use AdamW with weight decay
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        
        # OneCycle scheduler
        scheduler = get_scheduler(
            optimizer,
            scheduler_name='onecycle',
            num_epochs=50,
            steps_per_epoch=100  # Will be updated during training
        )
        
        return optimizer, scheduler
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training step with label smoothing."""
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': predicted
        }
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model specific arguments."""
        parser = parent_parser.add_argument_group("CrossAttention")
        parser.add_argument("--dropout_rate", type=float, default=0.3)
        parser.add_argument("--label_smoothing", type=float, default=0.1)
        return parent_parser 