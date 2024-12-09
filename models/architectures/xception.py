import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Tuple
from ..base_model import BaseModel
from ...utils.training import get_optimizer, get_scheduler, LabelSmoothingLoss

class XceptionModel(BaseModel):
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
        
        # Load pre-trained Xception
        self.model = timm.create_model(
            'xception',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            drop_rate=dropout_rate
        )
        
        # Additional feature processing
        self.feature_norm = nn.LayerNorm(self.model.num_features)
        self.feature_dropout = nn.Dropout(dropout_rate)
        
        # Multi-scale feature aggregation
        self.conv1x1 = nn.Conv2d(2048, 512, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Label smoothing loss
        self.criterion = LabelSmoothingLoss(num_classes, smoothing=label_smoothing)
        
        # Initialize weights
        self._initialize_weights()
        
        # Save hyperparameters
        self.save_hyperparameters = {
            'architecture': 'xception',
            'pretrained': pretrained,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate,
            'label_smoothing': label_smoothing
        }
        
    def _initialize_weights(self):
        """Initialize the weights of added layers."""
        for m in [self.conv1x1, self.classifier]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale feature aggregation."""
        # Get features from Xception
        features = self.model.forward_features(x)
        
        # Multi-scale processing
        conv_features = self.conv1x1(features)
        gap_features = self.gap(conv_features).view(x.size(0), -1)
        gmp_features = self.gmp(conv_features).view(x.size(0), -1)
        
        # Combine features
        combined = torch.cat([gap_features, gmp_features], dim=1)
        
        # Apply normalization and dropout
        combined = self.feature_norm(combined)
        combined = self.feature_dropout(combined)
        
        # Classification
        return self.classifier(combined)
        
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizer with layer-wise learning rates."""
        # Group parameters by layers
        backbone_params = list(self.model.parameters())
        new_params = list(self.conv1x1.parameters()) + \
                    list(self.classifier.parameters()) + \
                    list(self.feature_norm.parameters())
                    
        param_groups = [
            {'params': backbone_params, 'lr': 1e-4},    # Lower LR for pretrained
            {'params': new_params, 'lr': 3e-4}          # Higher LR for new layers
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
        parser = parent_parser.add_argument_group("Xception")
        parser.add_argument("--dropout_rate", type=float, default=0.3)
        parser.add_argument("--label_smoothing", type=float, default=0.1)
        return parent_parser 