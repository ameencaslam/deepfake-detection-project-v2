"""Model architectures package."""

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

def get_model(architecture: str, **kwargs):
    """Get model from registry by name."""
    if architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Available architectures: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[architecture](**kwargs)

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