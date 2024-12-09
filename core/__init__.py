"""Core functionality for the deepfake detection system."""

from .setup import setup_environment, setup_logging, setup_hardware
from .config import load_config, merge_cli_config
from .models import load_model, get_available_models

__all__ = [
    'setup_environment',
    'setup_logging',
    'setup_hardware',
    'load_config',
    'merge_cli_config',
    'load_model',
    'get_available_models'
] 