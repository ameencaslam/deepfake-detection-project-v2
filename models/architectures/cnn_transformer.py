import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Tuple, List, Optional
from models.base_model import BaseModel
from utils.training import get_optimizer, get_scheduler, LabelSmoothingLoss

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP block
        x = x + self.mlp(self.norm2(x))
        return x

class CNNTransformerModel(BaseModel):
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
        
        # CNN backbone (ResNet50) without classifier
        self.cnn = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=0,
            features_only=True
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.cnn(dummy_input)[-1]  # Get last layer features
            self.feature_dim = features.shape[1]
        
        # Feature processing
        self.conv1x1 = nn.Conv2d(self.feature_dim, 512, kernel_size=1)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        
        # Transformer blocks
        transformer_dim = 512
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, transformer_dim))  # 14x14 feature map
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(transformer_dim, dropout=dropout_rate)
            for _ in range(3)
        ])
        
        # Feature reduction
        self.feature_reduction = nn.Sequential(
            nn.Linear(transformer_dim, 512),
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
            'architecture': 'cnn_transformer',
            'pretrained': pretrained,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate,
            'label_smoothing': label_smoothing,
            'feature_dim': self.feature_dim
        }
        
    def _initialize_weights(self):
        """Initialize added layers."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in [self.conv1x1, self.feature_reduction, self.classifier]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining CNN and Transformer features."""
        # Get CNN features
        features = self.cnn(x)[-1]  # Get last layer features
        
        # Process features
        features = self.conv1x1(features)
        features = self.bn(features)
        features = self.relu(features)
        
        # Reshape for transformer
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Add positional embeddings
        features = features + self.pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            features = block(features)
            
        # Global average pooling
        features = features.mean(dim=1)
        
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
        transformer_params = list(self.transformer_blocks.parameters()) + \
                           [self.pos_embed]
        new_params = list(self.conv1x1.parameters()) + \
                    list(self.bn.parameters()) + \
                    list(self.feature_reduction.parameters()) + \
                    list(self.classifier.parameters()) + \
                    list(self.feature_norm.parameters())
                    
        param_groups = [
            {'params': cnn_params, 'lr': 1e-4},        # Lower LR for pretrained CNN
            {'params': transformer_params, 'lr': 3e-4}, # Higher LR for transformer
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
        parser = parent_parser.add_argument_group("CNNTransformer")
        parser.add_argument("--dropout_rate", type=float, default=0.3)
        parser.add_argument("--label_smoothing", type=float, default=0.1)
        return parent_parser 