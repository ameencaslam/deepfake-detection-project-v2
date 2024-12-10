import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Tuple
from models.base_model import BaseModel
from utils.training import get_optimizer, get_scheduler, LabelSmoothingLoss
import logging

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
        
        # Load ResNet without pretrained weights initially
        self.model = timm.create_model(
            'resnet50',
            pretrained=False,  # Always initialize without pretrained weights
            num_classes=0,  # Remove classifier
            drop_rate=dropout_rate
        )
        
        # Load pretrained weights if requested
        if pretrained:
            logging.info("Loading pretrained backbone weights...")
            pretrained_model = timm.create_model('resnet50', pretrained=True)
            # Only load backbone weights, not the classifier
            backbone_state_dict = {k: v for k, v in pretrained_model.state_dict().items() 
                                 if not k.startswith('fc')}
            self.model.load_state_dict(backbone_state_dict, strict=False)
            logging.info("Pretrained backbone weights loaded")
        else:
            logging.info("Initializing model without pretrained weights")
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.model.forward_features(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Feature processing
        self.feature_norm = nn.LayerNorm(512)
        self.feature_dropout = nn.Dropout(dropout_rate)
        
        # Simple transformer layer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        
        # Feature reduction
        self.feature_reduction = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
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
            'architecture': 'cnn_transformer',
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
        """Forward pass combining CNN and Transformer."""
        # Get CNN features
        features = self.model.forward_features(x)  # [B, C, H, W]
        
        # Reshape for transformer
        B, C, H, W = features.shape
        features = features.view(B, C, -1).transpose(1, 2)  # [B, HW, C]
        
        # Apply transformer
        features = self.transformer(features)
        
        # Global average pooling
        features = features.mean(dim=1)  # [B, C]
        
        # Feature processing
        features = self.feature_reduction(features)
        features = self.feature_norm(features)
        features = self.feature_dropout(features)
        
        # Classification
        return self.classifier(features)
        
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizer with layer-wise learning rates."""
        # Group parameters
        backbone_params = list(self.model.parameters())
        transformer_params = list(self.transformer.parameters())
        new_params = list(self.feature_reduction.parameters()) + \
                    list(self.classifier.parameters()) + \
                    list(self.feature_norm.parameters())
                    
        param_groups = [
            {'params': backbone_params, 'lr': 1e-4},        # Lower LR for pretrained
            {'params': transformer_params, 'lr': 3e-4},     # Higher LR for transformer
            {'params': new_params, 'lr': 3e-4}             # Higher LR for new layers
        ]
        
        # Get optimizer with proper weight decay
        optimizer = get_optimizer(
            self,
            optimizer_name='adamw',
            learning_rate=1e-4,
            weight_decay=0.01
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