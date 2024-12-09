import os

# Main paths - only these need to be changed if needed
DATASET_ROOT = '/content/3body-filtered-v2-10k'
PROJECT_ROOT = '/content/PROJECT-V2'

# Derived paths - automatically set
REAL_PATH = os.path.join(DATASET_ROOT, 'real')
FAKE_PATH = os.path.join(DATASET_ROOT, 'fake')

# Project paths
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
LOGS_PATH = os.path.join(PROJECT_ROOT, 'logs')
CHECKPOINTS_PATH = os.path.join(PROJECT_ROOT, 'checkpoints')

# Drive paths
DRIVE_PATH = '/content/drive/MyDrive/deepfake-project'
DRIVE_CHECKPOINTS = os.path.join(DRIVE_PATH, 'checkpoints')
DRIVE_RESULTS = os.path.join(DRIVE_PATH, 'results')

# Available models
MODELS = {
    'efficientnet': 'efficientnet',
    'swin': 'swin',
    'two_stream': 'two_stream',
    'xception': 'xception',
    'cnn_transformer': 'cnn_transformer',
    'cross_attention': 'cross_attention'
}

def setup_paths():
    """Create necessary directories."""
    paths = [
        MODELS_PATH, RESULTS_PATH, 
        LOGS_PATH, CHECKPOINTS_PATH
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)
        
    # Create Drive directories if Drive is mounted
    if os.path.exists('/content/drive'):
        os.makedirs(DRIVE_CHECKPOINTS, exist_ok=True)
        os.makedirs(DRIVE_RESULTS, exist_ok=True)
        
def validate_dataset():
    """Validate dataset exists."""
    required_paths = [
        (DATASET_ROOT, "Dataset root"),
        (REAL_PATH, "Real images"),
        (FAKE_PATH, "Fake images")
    ]
    
    for path, desc in required_paths:
        if not os.path.exists(path):
            raise ValueError(f"{desc} not found at: {path}")
        
def get_model_name(model_key: str) -> str:
    """Get full model name from key."""
    if model_key not in MODELS:
        valid_models = list(MODELS.keys()) + ['all']
        raise ValueError(f"Invalid model: {model_key}. Choose from: {valid_models}")
    return MODELS[model_key]

def get_checkpoint_dir(model_name: str = None) -> str:
    """Get checkpoint directory for a model."""
    if model_name:
        return os.path.join(CHECKPOINTS_PATH, model_name)
    return CHECKPOINTS_PATH

def get_data_dir() -> str:
    """Get data directory."""
    return DATASET_ROOT