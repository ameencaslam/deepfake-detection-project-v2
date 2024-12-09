"""Setup utilities for environment, logging, and hardware."""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from utils.hardware import HardwareManager
from config.base_config import Config

def setup_environment(config: Config) -> None:
    """Setup the environment including directories and dependencies."""
    # Create necessary directories
    for dir_path in [
        config.paths.checkpoints,
        config.paths.logs,
        config.paths.results,
        config.paths.cache
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def setup_logging(log_dir: Optional[str] = None, name: str = 'deepfake') -> logging.Logger:
    """Setup unified logging across the application."""
    logger = logging.getLogger(name)
    
    if logger.handlers:  # Return existing logger if already setup
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = Path(log_dir) / f'{name}_{timestamp}.log'
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def setup_hardware(config: Config) -> HardwareManager:
    """Setup and return hardware manager with given configuration."""
    return HardwareManager(
        device=config.hardware.device,
        memory_fraction=config.hardware.gpu_memory_fraction,
        seed=config.training.seed
    ) 