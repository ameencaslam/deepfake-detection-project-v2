import torch
import psutil
import gc
from typing import Dict, Any

def get_device() -> str:
    """Get the device to use for training (cuda or cpu)."""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

class HardwareManager:
    def __init__(self):
        """Initialize hardware manager."""
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        
        # Get GPU info if available
        if self.gpu_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
        else:
            self.gpu_name = None
            self.gpu_memory_total = None
            
        # Set up mixed precision
        self.mixed_precision = self._setup_mixed_precision()
        
        # Set up memory optimization
        self._setup_memory_optimization()
        
    def _setup_mixed_precision(self) -> bool:
        """Set up mixed precision training."""
        if self.gpu_available and torch.cuda.is_available():
            return True
        return False
        
    def _setup_memory_optimization(self):
        """Set up memory optimization settings."""
        if self.gpu_available:
            # Empty CUDA cache
            torch.cuda.empty_cache()
            # Run garbage collector
            gc.collect()
            
    def get_hardware_stats(self) -> Dict[str, float]:
        """Get current hardware statistics."""
        stats = {
            'cpu_utilization': psutil.cpu_percent(),
            'ram_used': psutil.virtual_memory().used / 1e9,  # Convert to GB
            'ram_total': psutil.virtual_memory().total / 1e9  # Convert to GB
        }
        
        if self.gpu_available:
            stats.update({
                'gpu_memory_used': torch.cuda.memory_allocated(0) / 1e9,  # Convert to GB
                'gpu_memory_total': self.gpu_memory_total
            })
            
        return stats
        
    def print_hardware_info(self):
        """Print hardware configuration information."""
        print("Hardware Configuration:")
        print(f"├── Device: {self.device.type.upper()}")
        if self.gpu_available:
            print(f"├── GPU: {self.gpu_name}")
            print(f"├── GPU Memory: {self.gpu_memory_total:.1f}GB")
            print(f"├── CUDA Version: {torch.version.cuda}")
        print(f"├── Mixed Precision: {'Enabled' if self.mixed_precision else 'Disabled'}")
        print(f"├── CPU Usage: {psutil.cpu_percent()}%")
        ram = psutil.virtual_memory()
        print(f"└── RAM Usage: {ram.used/1e9:.1f}GB/{ram.total/1e9:.1f}GB")
        
    def optimize_memory(self):
        """Optimize memory usage."""
        if self.gpu_available:
            torch.cuda.empty_cache()
        gc.collect()