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
        
        # Frequency stream
        self.frequency_stream = FrequencyBranch(in_channels=3)
        
        # Feature fusion
        spatial_features = self.spatial_stream.num_features
        freq_features = 256 * 28 * 28  # From FrequencyBranch output
        
        self.fusion = nn.Sequential(
            nn.Linear(spatial_features + freq_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Label smoothing loss
        self.criterion = LabelSmoothingLoss(num_classes, smoothing=label_smoothing)
        
        # Save hyperparameters
        self.save_hyperparameters = {
            'architecture': 'two_stream',
            'pretrained': pretrained,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate,
            'label_smoothing': label_smoothing
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining spatial and frequency streams."""
        # Spatial stream
        spatial_features = self.spatial_stream.forward_features(x)
        
        # Frequency stream
        freq_features = self.frequency_stream(x)
        freq_features = freq_features.view(freq_features.size(0), -1)
        
        # Combine features
        combined = torch.cat([spatial_features, freq_features], dim=1)
        
        # Final classification
        return self.fusion(combined)
        
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizer with different learning rates for streams."""
        # Parameters with different learning rates
        spatial_params = list(self.spatial_stream.parameters())
        other_params = list(self.frequency_stream.parameters()) + list(self.fusion.parameters())
        
        param_groups = [
            {'params': spatial_params, 'lr': 1e-4},  # Lower LR for pretrained
            {'params': other_params, 'lr': 3e-4}     # Higher LR for new parts
        ]
        
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