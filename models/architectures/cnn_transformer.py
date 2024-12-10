import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Tuple, Optional
from models.base_model import BaseModel
from utils.training import get_optimizer, get_scheduler, LabelSmoothingLoss
import logging

class CNNTransformerModel(BaseModel):
    """Hybrid model combining CNN backbone with transformer layers."""
    
    def __init__(self,
                 pretrained: bool = True,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3,
                 label_smoothing: float = 0.1,
                 transformer_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 **kwargs):
        config = {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'mixed_precision': True
        }
        super().__init__(config)
        
        # Load CNN backbone without pretrained weights initially
        self.cnn = timm.create_model(
            'resnet50',
            pretrained=False,
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
            self.cnn.load_state_dict(backbone_state_dict, strict=False)
            logging.info("Pretrained backbone weights loaded")
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.cnn.forward_features(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Feature processing
        self.conv1x1 = nn.Conv2d(self.feature_dim, transformer_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(transformer_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(512)
        self.feature_dropout = nn.Dropout(dropout_rate)
        
        # Feature reduction
        self.feature_reduction = nn.Sequential(
            nn.Linear(transformer_dim, 512),
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
        
        # Loss function
        self.criterion = LabelSmoothingLoss(num_classes, smoothing=label_smoothing)
        
        # Initialize weights
        self._initialize_weights()
        
        # Save hyperparameters
        self.hparams = {
            'architecture': 'cnn_transformer',
            'pretrained': pretrained,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate,
            'label_smoothing': label_smoothing,
            'transformer_dim': transformer_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'feature_dim': self.feature_dim
        }
        
    def _initialize_weights(self):
        """Initialize model weights."""
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
        """Forward pass of the model."""
        # CNN feature extraction
        features = self.cnn.forward_features(x)  # [B, C, H, W]
        
        # Process features
        features = self.conv1x1(features)  # [B, transformer_dim, H, W]
        features = self.bn(features)
        features = self.relu(features)
        
        # Reshape for transformer
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # [B, HW, C]
        
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
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single training step."""
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
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
        
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single validation step."""
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
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
        
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        """Configure optimizer and scheduler."""
        # Group parameters
        cnn_params = list(self.cnn.parameters())
        transformer_params = list(self.transformer.parameters())
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
        
    def get_model_specific_args(self) -> Dict[str, Any]:
        """Get model-specific arguments."""
        return {
            'dropout_rate': 0.3,
            'label_smoothing': 0.1,
            'transformer_dim': 512,
            'num_heads': 8,
            'num_layers': 3
        }