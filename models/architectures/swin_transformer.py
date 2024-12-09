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
        
        # Load pre-trained Swin Transformer
        self.model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
            drop_path_rate=0.2,  # Stochastic depth
        )
        
        # Add additional normalization and dropout
        self.norm1 = nn.LayerNorm(self.model.num_features)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(self.model.num_features)
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)  # Lower dropout near output
        
        # Label smoothing loss
        self.criterion = LabelSmoothingLoss(num_classes, smoothing=label_smoothing)
        
        # Save hyperparameters
        self.save_hyperparameters = {
            'architecture': 'swin_transformer',
            'pretrained': pretrained,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate,
            'label_smoothing': label_smoothing
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with additional normalization and dropout."""
        # Get features from Swin Transformer
        features = self.model.forward_features(x)
        
        # Apply additional regularization
        features = self.norm1(features)
        features = self.dropout1(features)
        
        # Global average pooling
        features = features.mean(dim=1)
        
        # Final normalization and classification
        features = self.norm2(features)
        features = self.dropout2(features)
        
        return self.model.head(features)
        
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizer and learning rate scheduler."""
        # Get optimizer with proper weight decay
        optimizer = get_optimizer(
            self,
            optimizer_name='adamw',  # Better for transformers
            learning_rate=1e-4,
            weight_decay=0.05  # Higher weight decay for transformers
        )
        
        # Get OneCycle scheduler
        scheduler = get_scheduler(
            optimizer,
            scheduler_name='onecycle',
            num_epochs=50,
            steps_per_epoch=100  # This will be updated in training
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