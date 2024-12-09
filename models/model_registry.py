import torch
from typing import Dict, Any, Optional, Type, List, Union
from pathlib import Path
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
import importlib
import inspect

from .architectures.efficientnet_b3 import EfficientNetModel
from .architectures.swin_transformer import SwinTransformerModel
from .architectures.two_stream import TwoStreamModel
from .architectures.xception import XceptionModel
from .architectures.cnn_transformer import CNNTransformerModel
from .architectures.cross_attention import CrossAttentionModel
from .architectures.vision_transformer import VisionTransformerModel
from .architectures.convnext import ConvNeXTModel
from .base_model import BaseModel

@dataclass
class ModelMetadata:
    """Metadata for registered models."""
    name: str
    version: str
    description: str
    architecture: str
    paper_link: Optional[str] = None
    default_config: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    performance: Dict[str, float] = field(default_factory=dict)

class ModelRegistry:
    """Enhanced model registry with versioning and metadata."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._models: Dict[str, Type[BaseModel]] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
        
        # Register default models
        self._register_default_models()
        
    def _register_default_models(self):
        """Register default model architectures."""
        default_models = {
            'efficientnet': {
                'class': EfficientNetModel,
                'metadata': ModelMetadata(
                    name='EfficientNet-B3',
                    version='1.0.0',
                    description='EfficientNet model with attention and feature visualization',
                    architecture='efficientnet_b3',
                    paper_link='https://arxiv.org/abs/1905.11946',
                    default_config={
                        'pretrained': True,
                        'dropout_rate': 0.3,
                        'gradient_checkpointing': False
                    },
                    requirements=['timm>=0.9.0']
                )
            },
            'swin': {
                'class': SwinTransformerModel,
                'metadata': ModelMetadata(
                    name='Swin Transformer',
                    version='1.0.0',
                    description='Swin Transformer with hierarchical feature fusion',
                    architecture='swin_transformer',
                    paper_link='https://arxiv.org/abs/2103.14030',
                    default_config={
                        'pretrained': True,
                        'dropout_rate': 0.3,
                        'window_size': 7
                    },
                    requirements=['timm>=0.9.0']
                )
            },
            'two_stream': {
                'class': TwoStreamModel,
                'metadata': ModelMetadata(
                    name='Two-Stream Network',
                    version='1.0.0',
                    description='Two-stream architecture for spatial and frequency analysis',
                    architecture='two_stream',
                    default_config={
                        'pretrained': True,
                        'dropout_rate': 0.3,
                        'fusion_type': 'concat'
                    }
                )
            },
            'xception': {
                'class': XceptionModel,
                'metadata': ModelMetadata(
                    name='Xception',
                    version='1.0.0',
                    description='Modified Xception architecture for deepfake detection',
                    architecture='xception',
                    paper_link='https://arxiv.org/abs/1610.02357',
                    default_config={
                        'pretrained': True,
                        'dropout_rate': 0.3
                    }
                )
            },
            'cnn_transformer': {
                'class': CNNTransformerModel,
                'metadata': ModelMetadata(
                    name='CNN-Transformer',
                    version='1.0.0',
                    description='Hybrid CNN-Transformer architecture',
                    architecture='cnn_transformer',
                    default_config={
                        'pretrained': True,
                        'dropout_rate': 0.3,
                        'num_heads': 8
                    }
                )
            },
            'cross_attention': {
                'class': CrossAttentionModel,
                'metadata': ModelMetadata(
                    name='Cross-Attention',
                    version='1.0.0',
                    description='Cross-attention model for feature interaction',
                    architecture='cross_attention',
                    default_config={
                        'pretrained': True,
                        'dropout_rate': 0.3,
                        'num_heads': 8
                    }
                )
            },
            'vit': {
                'class': VisionTransformerModel,
                'metadata': ModelMetadata(
                    name='Vision Transformer',
                    version='1.0.0',
                    description='Vision Transformer with attention visualization',
                    architecture='vit',
                    paper_link='https://arxiv.org/abs/2010.11929',
                    default_config={
                        'pretrained': True,
                        'dropout_rate': 0.3,
                        'patch_size': 16
                    },
                    requirements=['timm>=0.9.0']
                )
            },
            'convnext': {
                'class': ConvNeXTModel,
                'metadata': ModelMetadata(
                    name='ConvNeXT',
                    version='1.0.0',
                    description='ConvNeXT architecture with modern CNN design',
                    architecture='convnext',
                    paper_link='https://arxiv.org/abs/2201.03545',
                    default_config={
                        'pretrained': True,
                        'dropout_rate': 0.3
                    },
                    requirements=['timm>=0.9.0']
                )
            }
        }
        
        for name, info in default_models.items():
            self.register_model(name, info['class'], info['metadata'])
            
    def register_model(self, name: str, model_class: Type[BaseModel],
                      metadata: Optional[ModelMetadata] = None):
        """Register a new model architecture."""
        # Validate model class
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class must inherit from BaseModel: {model_class}")
            
        # Validate required methods
        required_methods = ['forward', 'get_features']
        for method in required_methods:
            if not hasattr(model_class, method):
                raise ValueError(f"Model class must implement {method} method")
                
        # Create default metadata if not provided
        if metadata is None:
            metadata = ModelMetadata(
                name=model_class.__name__,
                version='1.0.0',
                description='Custom model architecture',
                architecture=name.lower()
            )
            
        # Register model
        self._models[name] = model_class
        self._metadata[name] = metadata
        self.logger.info(f"Registered model: {name} (version {metadata.version})")
        
    def get_model(self, name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Get model instance by name."""
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}. Available models: {list(self._models.keys())}")
            
        # Get model class and metadata
        model_class = self._models[name]
        metadata = self._metadata[name]
        
        # Merge default config with provided config
        if config is None:
            config = {}
        merged_config = {**metadata.default_config, **config}
        
        # Create model instance
        try:
            model = model_class(merged_config)
            return model
        except Exception as e:
            self.logger.error(f"Error creating model {name}: {str(e)}")
            raise
            
    def list_models(self) -> List[ModelMetadata]:
        """List all registered models with metadata."""
        return list(self._metadata.values())
        
    def get_model_info(self, name: str) -> ModelMetadata:
        """Get detailed information about a model."""
        if name not in self._metadata:
            raise ValueError(f"Unknown model: {name}")
        return self._metadata[name]
        
    def update_model_performance(self, name: str, metrics: Dict[str, float]):
        """Update model performance metrics."""
        if name not in self._metadata:
            raise ValueError(f"Unknown model: {name}")
            
        self._metadata[name].performance.update(metrics)
        
    def save_registry(self, path: Union[str, Path]):
        """Save registry metadata to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert metadata to dictionary
        registry_data = {
            name: metadata.__dict__
            for name, metadata in self._metadata.items()
        }
        
        with open(path, 'w') as f:
            json.dump(registry_data, f, indent=4)
            
    def load_registry(self, path: Union[str, Path]):
        """Load registry metadata from file."""
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Registry file not found: {path}")
            
        with open(path, 'r') as f:
            registry_data = json.load(f)
            
        # Update metadata
        for name, data in registry_data.items():
            if name in self._metadata:
                self._metadata[name] = ModelMetadata(**data)
                
    def get_model_requirements(self, name: str) -> List[str]:
        """Get model requirements."""
        if name not in self._metadata:
            raise ValueError(f"Unknown model: {name}")
            
        return self._metadata[name].requirements
        
    def validate_model_requirements(self, name: str) -> bool:
        """Validate that model requirements are met."""
        requirements = self.get_model_requirements(name)
        
        for req in requirements:
            package_name = req.split('>=')[0]
            try:
                importlib.import_module(package_name)
            except ImportError:
                return False
        return True
        
    def get_model_config_schema(self, name: str) -> Dict[str, Any]:
        """Get model configuration schema."""
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}")
            
        model_class = self._models[name]
        init_signature = inspect.signature(model_class.__init__)
        
        schema = {}
        for param_name, param in init_signature.parameters.items():
            if param_name not in ['self', 'args', 'kwargs']:
                schema[param_name] = {
                    'type': str(param.annotation),
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'required': param.default == inspect.Parameter.empty
                }
                
        return schema
        
    def __str__(self) -> str:
        """String representation of registry."""
        lines = ["Model Registry:"]
        for name, metadata in self._metadata.items():
            lines.append(f"\n{metadata.name} (version {metadata.version})")
            lines.append(f"Architecture: {metadata.architecture}")
            lines.append(f"Description: {metadata.description}")
            if metadata.paper_link:
                lines.append(f"Paper: {metadata.paper_link}")
            if metadata.performance:
                lines.append("Performance:")
                for metric, value in metadata.performance.items():
                    lines.append(f"  {metric}: {value:.4f}")
                    
        return "\n".join(lines)

# Create global registry instance
registry = ModelRegistry()

# Convenience functions
def get_model(name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
    """Get model instance by name."""
    return registry.get_model(name, config)

def list_models() -> List[ModelMetadata]:
    """List all registered models."""
    return registry.list_models()

def register_model(name: str, model_class: Type[BaseModel],
                  metadata: Optional[ModelMetadata] = None):
    """Register a new model."""
    registry.register_model(name, model_class, metadata) 