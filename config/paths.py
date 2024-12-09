import os

# Main paths - only these need to be changed if needed
DATASET_ROOT = '/content/3body-filtered-v2-10k'
PROJECT_ROOT = '/content/deepfake_project'

# Derived paths - automatically set
REAL_PATH = os.path.join(DATASET_ROOT, 'real')
FAKE_PATH = os.path.join(DATASET_ROOT, 'fake')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
LOGS_PATH = os.path.join(PROJECT_ROOT, 'logs')
DRIVE_PATH = '/content/drive/MyDrive/deepfake_project'

# Available models
MODELS = {
    'swin': 'swin_transformer',
    'two_stream': 'two_stream',
    'xception': 'xception',
    'cnn_transformer': 'cnn_transformer',
    'cross_attention': 'cross_attention'
}

def setup_paths():
    """Create necessary directories."""
    paths = [PROJECT_ROOT, MODELS_PATH, RESULTS_PATH, LOGS_PATH]
    for path in paths:
        os.makedirs(path, exist_ok=True)
        
def validate_dataset():
    """Validate dataset exists."""
    if not os.path.exists(DATASET_ROOT):
        raise ValueError(f"Dataset not found at: {DATASET_ROOT}")
    if not os.path.exists(REAL_PATH):
        raise ValueError(f"Real images not found at: {REAL_PATH}")
    if not os.path.exists(FAKE_PATH):
        raise ValueError(f"Fake images not found at: {FAKE_PATH}")
        
def get_model_name(model_key: str) -> str:
    """Get full model name from key."""
    if model_key not in MODELS:
        valid_models = list(MODELS.keys()) + ['all']
        raise ValueError(f"Invalid model: {model_key}. Choose from: {valid_models}")
    return MODELS[model_key] 