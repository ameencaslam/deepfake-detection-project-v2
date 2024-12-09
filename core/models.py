"""Model management utilities."""

from typing import Dict, Any, Optional
import torch
from config.base_config import Config
from models.model_registry import registry as model_registry

def get_available_models() -> Dict[str, str]:
    """Get dictionary of available models and their descriptions."""
    return {
        'swin': 'Swin Transformer',
        'two_stream': 'Two-Stream Network',
        'xception': 'Xception',
        'cnn_transformer': 'CNN-Transformer Hybrid',
        'cross_attention': 'Cross-Attention Model',
        'efficientnet': 'EfficientNet-B3'
    }

def load_model(config: Config) -> torch.nn.Module:
    """Load model based on configuration."""
    model_name = config.model.architecture
    if model_name not in model_registry:
        raise ValueError(f"Model {model_name} not found in registry")
    
    # Get model class and create instance
    model_class = model_registry[model_name]
    model = model_class(**config.model.parameters)
    
    # Load checkpoint if specified
    if hasattr(config.model, 'checkpoint_path'):
        checkpoint = torch.load(
            config.model.checkpoint_path,
            map_location=config.hardware.device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model 