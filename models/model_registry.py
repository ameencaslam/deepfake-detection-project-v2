from .architectures.swin_transformer import SwinTransformerModel
from .architectures.two_stream import TwoStreamModel
from .architectures.xception import XceptionModel
from .architectures.cnn_transformer import CNNTransformerModel
from .architectures.cross_attention import CrossAttentionModel
from .architectures.efficientnet_b3 import EfficientNetModel

MODEL_REGISTRY = {
    'swin': SwinTransformerModel,
    'two_stream': TwoStreamModel,
    'xception': XceptionModel,
    'cnn_transformer': CNNTransformerModel,
    'cross_attention': CrossAttentionModel,
    'efficientnet': EfficientNetModel
}

def get_model(model_name: str, **kwargs):
    """Get model from registry by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs) 