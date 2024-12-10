import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Tuple, List, Optional
from models.base_model import BaseModel
from utils.training import get_optimizer, get_scheduler, LabelSmoothingLoss
import torch.fft as fft

class FrequencyBranch(nn.Module):
    """Frequency domain processing branch."""
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels*2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to frequency domain
        ffted = fft.fft2(x, dim=(-2, -1))
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)
        
        # Process frequency information
        x = self.relu(self.bn1(self.conv1(ffted)))
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        return x

class TwoStreamModel(BaseModel):
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
        
        # Spatial stream (using EfficientNet)
        self.spatial_stream = timm.create_model(
            'efficientnet_b0',  # Lighter model for spatial stream
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            drop_rate=dropout_rate
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            spatial_features = self.spatial_stream.forward_features(dummy_input)
            self.spatial_dim = spatial_features.shape[1]
        
        # Frequency stream
        self.frequency_stream = FrequencyBranch(in_channels=3)
        
        # Multi-scale pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Feature processing
        total_features = self.spatial_dim + 256  # spatial + frequency features
        self.feature_reduction = nn.Sequential(
            nn.Linear(total_features * 2, 512),  # *2 for concatenated pooling features
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
            'architecture': 'two_stream',
            'pretrained': pretrained,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate,
            'label_smoothing': label_smoothing,
            'spatial_dim': self.spatial_dim
        }
        
    def _initialize_weights(self):
        """Initialize the weights of added layers."""
        for m in [self.feature_reduction, self.classifier]:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining spatial and frequency streams."""
        # Spatial stream
        spatial_features = self.spatial_stream.forward_features(x)  # [B, C]
        
        # Frequency stream
        freq_features = self.frequency_stream(x)  # [B, C, H, W]
        
        # Multi-scale pooling for both streams
        spatial_avg = self.global_pool(spatial_features.unsqueeze(-1).unsqueeze(-1)).flatten(1)
        spatial_max = self.max_pool(spatial_features.unsqueeze(-1).unsqueeze(-1)).flatten(1)
        
        freq_avg = self.global_pool(freq_features).flatten(1)
        freq_max = self.max_pool(freq_features).flatten(1)
        
        # Combine features
        combined = torch.cat([spatial_avg, spatial_max, freq_avg, freq_max], dim=1)
        
        # Feature processing
        features = self.feature_reduction(combined)
        features = self.feature_norm(features)
        features = self.feature_dropout(features)
        
        # Classification
        return self.classifier(features)
        
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizer with layer-wise learning rates."""
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
        parser = parent_parser.add_argument_group("TwoStream")
        parser.add_argument("--dropout_rate", type=float, default=0.3)
        parser.add_argument("--label_smoothing", type=float, default=0.1)
        return parent_parser 