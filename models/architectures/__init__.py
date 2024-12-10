"""Model architectures package."""

from typing import Dict, Any
from models.architectures.efficientnet_b3 import EfficientNetModel
from models.architectures.swin_transformer import SwinTransformerModel
from models.architectures.two_stream import TwoStreamModel
from models.architectures.xception import XceptionModel
from models.architectures.cnn_transformer import CNNTransformerModel
from models.architectures.cross_attention import CrossAttentionModel

MODEL_REGISTRY = {
    'efficientnet': EfficientNetModel,
    'swin': SwinTransformerModel,
    'two_stream': TwoStreamModel,
    'xception': XceptionModel,
    'cnn_transformer': CNNTransformerModel,
    'cross_attention': CrossAttentionModel
}

def get_model(architecture: str, **kwargs) -> Any:
    """Get model instance by architecture name.
    
    Args:
        architecture: Name of the model architecture
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If architecture is not supported
    """
    if architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"Architecture {architecture} not supported. "
            f"Available architectures: {list(MODEL_REGISTRY.keys())}"
        )
    
    # Create model instance
    model_class = MODEL_REGISTRY[architecture]
    return model_class(**kwargs)

__all__ = [
    'EfficientNetModel',
    'SwinTransformerModel',
    'TwoStreamModel',
    'XceptionModel',
    'CNNTransformerModel',
    'CrossAttentionModel',
    'get_model',
    'MODEL_REGISTRY'
] 