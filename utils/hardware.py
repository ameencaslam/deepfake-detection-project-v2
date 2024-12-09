import torch
import psutil
import os
from typing import Dict, Any, Optional
import gc
from dataclasses import dataclass

@dataclass
class HardwareStats:
    gpu_utilization: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    cpu_utilization: float = 0.0
    ram_used: float = 0.0
    ram_total: float = 0.0

def get_device() -> str:
    """Get the device to use for training (cuda or cpu)."""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

class HardwareManager:
    def __init__(self):
        """Initialize hardware manager."""
        self.device = get_device()
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
        else:
            self.gpu_name = None
            self.gpu_memory_total = None
        
        self.mixed_precision = self._setup_mixed_precision()
        self._setup_memory_optimization()
        
    def _get_device(self) -> torch.device:
        """Detect and return the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device("xpu")
        else:
            return torch.device("cpu")
            
    def _setup_mixed_precision(self) -> bool:
        """Setup mixed precision training if supported."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            return True
        return False
        
    def _setup_memory_optimization(self):
        """Apply memory optimizations based on device."""
        if self.device.type == "cuda":
            # Empty CUDA cache
            torch.cuda.empty_cache()
            # Set memory allocation to TensorFloat-32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Enable cudnn benchmarking for faster training
            torch.backends.cudnn.benchmark = True
            
    def get_hardware_stats(self) -> HardwareStats:
        """Get current hardware utilization statistics."""
        stats = HardwareStats()
        
        # CPU and RAM stats
        stats.cpu_utilization = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        stats.ram_used = ram.used / (1024 ** 3)  # Convert to GB
        stats.ram_total = ram.total / (1024 ** 3)
        
        # GPU stats if available
        if self.device.type == "cuda":
            try:
                gpu_stats = torch.cuda.get_device_properties(0)
                stats.gpu_memory_total = gpu_stats.total_memory / (1024 ** 3)
                stats.gpu_memory_used = (torch.cuda.memory_allocated() + 
                                       torch.cuda.memory_reserved()) / (1024 ** 3)
                stats.gpu_utilization = torch.cuda.utilization()
            except:
                pass
                
        return stats
        
    def optimize_memory(self):
        """Perform memory optimization."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        info = {
            'device_type': self.device.type,
            'mixed_precision': self.mixed_precision,
        }
        
        if self.device.type == "cuda":
            gpu_props = torch.cuda.get_device_properties(0)
            info.update({
                'gpu_name': gpu_props.name,
                'gpu_memory': f"{gpu_props.total_memory / (1024**3):.1f}GB",
                'cuda_version': torch.version.cuda,
            })
            
        return info
        
    def print_hardware_info(self):
        """Print detailed hardware information."""
        info = self.get_device_info()
        stats = self.get_hardware_stats()
        
        print("Hardware Configuration:")
        print(f"├── Device: {info['device_type'].upper()}")
        if info['device_type'] == 'cuda':
            print(f"├── GPU: {info['gpu_name']}")
            print(f"├── GPU Memory: {info['gpu_memory']}")
            print(f"├── CUDA Version: {info['cuda_version']}")
        print(f"├── Mixed Precision: {'Enabled' if info['mixed_precision'] else 'Disabled'}")
        print(f"├── CPU Usage: {stats.cpu_utilization:.1f}%")
        print(f"└── RAM Usage: {stats.ram_used:.1f}GB/{stats.ram_total:.1f}GB") 