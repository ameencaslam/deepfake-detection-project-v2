import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Tuple, List, Optional
from models.base_model import BaseModel
from utils.training import get_optimizer, get_scheduler, LabelSmoothingLoss

class SwinTransformerModel(BaseModel):
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
        
        # Load pre-trained Swin Transformer without classifier
        self.model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            drop_rate=dropout_rate,
            drop_path_rate=0.2,  # Stochastic depth
        )
        
        # Get feature dimensions
        with torch.no_grad():
            # Pass a dummy input to get feature dimensions
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.model.forward_features(dummy_input)
            self.feature_dim = features.shape[-1]  # Get feature dimension
        
        # Feature processing
        self.feature_norm = nn.LayerNorm(512)
        self.feature_dropout = nn.Dropout(dropout_rate)
        
        # Multi-scale feature aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Additional feature processing
        self.feature_reduction = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
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
            'architecture': 'swin_transformer',
            'pretrained': pretrained,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate,
            'label_smoothing': label_smoothing,
            'feature_dim': self.feature_dim
        }
        
    def _initialize_weights(self):
        """Initialize added layers."""
        for m in [self.feature_reduction, self.classifier]:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale feature aggregation."""
        # Get features from Swin Transformer
        features = self.model.forward_features(x)  # [B, N, C]
        
        # Multi-scale pooling (treating sequence dimension as channels)
        features = features.transpose(1, 2)  # [B, C, N]
        avg_features = self.global_pool(features).squeeze(-1)  # [B, C]
        max_features = self.max_pool(features).squeeze(-1)  # [B, C]
        
        # Combine features
        combined = torch.cat([avg_features, max_features], dim=1)
        
        # Feature processing
        features = self.feature_reduction(combined)
        features = self.feature_norm(features)
        features = self.feature_dropout(features)
        
        # Classification
        return self.classifier(features)
        
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizer with layer-wise learning rates."""
        # Group parameters
        backbone_params = list(self.model.parameters())
        new_params = list(self.feature_reduction.parameters()) + \
                    list(self.classifier.parameters()) + \
                    list(self.feature_norm.parameters())
                    
        param_groups = [
            {'params': backbone_params, 'lr': 1e-4},    # Lower LR for pretrained
            {'params': new_params, 'lr': 3e-4}          # Higher LR for new layers
        ]
        
        # Get optimizer with proper weight decay
        optimizer = get_optimizer(
            self,
            optimizer_name='adamw',  # Better for transformers
            learning_rate=1e-4,
            weight_decay=0.05  # Higher weight decay for transformers
        )
        
        # Get scheduler
        scheduler = get_scheduler(
            optimizer,
            scheduler_name='onecycle',
            num_epochs=50,
            steps_per_epoch=100  # Will be updated during training
        )
        
        return optimizer, scheduler
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single training step with label smoothing."""
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
        """Add model specific arguments to parser."""
        parser = parent_parser.add_argument_group("SwinTransformer")
        parser.add_argument("--dropout_rate", type=float, default=0.3)
        parser.add_argument("--label_smoothing", type=float, default=0.1)
        return parent_parser 