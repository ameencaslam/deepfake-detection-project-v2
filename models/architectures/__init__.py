from typing import Dict, Any
import torch.nn as nn
from .efficientnet import EfficientNetModel
from .swin_transformer import SwinTransformerModel
from .two_stream import TwoStreamModel
from .xception import XceptionModel
from .cnn_transformer import CNNTransformerModel
from .cross_attention import CrossAttentionModel

MODEL_REGISTRY = {
    'efficientnet_b3': EfficientNetModel,
    'swin_transformer': SwinTransformerModel,
    'two_stream': TwoStreamModel,
    'xception': XceptionModel,
    'cnn_transformer': CNNTransformerModel,
    'cross_attention': CrossAttentionModel
}

def get_model(architecture: str, **kwargs) -> nn.Module:
    """
    Factory function to get the specified model architecture.
    
    Args:
        architecture: Name of the model architecture
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        Initialized model
    """
    if architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Available architectures: {list(MODEL_REGISTRY.keys())}"
        )
        
    model_class = MODEL_REGISTRY[architecture]
    return model_class(**kwargs) 