import torch
import torch.nn as nn
import timm
import torch.fft as fft
from typing import Dict, Any, Tuple, Optional
from models.base_model import BaseModel
from utils.training import get_optimizer, get_scheduler, LabelSmoothingLoss
import logging

class TwoStreamModel(BaseModel):
    """Two-stream model combining spatial and frequency domain analysis."""
    
    def __init__(self,
                 pretrained: bool = True,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3,
                 label_smoothing: float = 0.1,
                 **kwargs):
        config = {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'mixed_precision': True
        }
        super().__init__(config)
        
        # Load spatial stream without pretrained weights initially
        self.spatial_stream = timm.create_model(
            'efficientnet_b0',
            pretrained=False,
            num_classes=0,  # Remove classifier
            drop_rate=dropout_rate
        )
        
        # Load pretrained weights if requested
        if pretrained:
            logging.info("Loading pretrained backbone weights...")
            pretrained_model = timm.create_model('efficientnet_b0', pretrained=True)
            # Only load backbone weights, not the classifier
            backbone_state_dict = {k: v for k, v in pretrained_model.state_dict().items() 
                                 if not k.startswith('classifier')}
            self.spatial_stream.load_state_dict(backbone_state_dict, strict=False)
            logging.info("Pretrained backbone weights loaded")
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.spatial_stream.forward_features(dummy_input)
            self.spatial_dim = features.shape[1]
        
        # Frequency stream
        self.frequency_stream = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),  # 6 channels for real and imaginary parts
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Feature processing
        self.feature_norm = nn.LayerNorm(512)
        self.feature_dropout = nn.Dropout(dropout_rate)
        
        # Feature reduction
        self.feature_reduction = nn.Sequential(
            nn.Linear(self.spatial_dim + 256, 512),  # Spatial + Frequency features
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
            'architecture': 'two_stream',
            'pretrained': pretrained,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate,
            'label_smoothing': label_smoothing,
            'spatial_dim': self.spatial_dim
        }
        
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in [self.frequency_stream, self.feature_reduction, self.classifier]:
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
        # Spatial stream
        spatial_features = self.spatial_stream.forward_features(x)  # [B, C]
        
        # Frequency stream
        ffted = fft.fft2(x, dim=(-2, -1))
        freq_input = torch.cat([ffted.real, ffted.imag], dim=1)  # Combine real and imaginary parts
        freq_features = self.frequency_stream(freq_input).flatten(1)  # [B, 256]
        
        # Combine features
        combined = torch.cat([spatial_features, freq_features], dim=1)
        
        # Feature processing
        features = self.feature_reduction(combined)
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
        spatial_params = list(self.spatial_stream.parameters())
        freq_params = list(self.frequency_stream.parameters())
        new_params = list(self.feature_reduction.parameters()) + \
                    list(self.classifier.parameters()) + \
                    list(self.feature_norm.parameters())
        
        param_groups = [
            {'params': spatial_params, 'lr': 1e-4},  # Lower LR for pretrained
            {'params': freq_params, 'lr': 3e-4},     # Higher LR for freq stream
            {'params': new_params, 'lr': 3e-4}       # Higher LR for new layers
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
            'freq_channels': 64,
            'freq_layers': 3
        }