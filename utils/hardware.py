import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from pathlib import Path
import json
import psutil
import GPUtil
import numpy as np
from dataclasses import dataclass, field
import os
import warnings
from contextlib import contextmanager
import time
import subprocess
from threading import Thread, Event
import queue

@dataclass
class HardwareStats:
    """Hardware statistics."""
    gpu_utilization: float
    gpu_memory_used: float
    gpu_memory_total: float
    cpu_utilization: float
    cpu_memory_used: float
    cpu_memory_total: float
    temperature: Optional[float] = None
    power_usage: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

class HardwareMonitor:
    """Monitors hardware utilization."""
    
    def __init__(self, interval: float = 1.0):
        """Initialize hardware monitor.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.stats_queue = queue.Queue()
        self.stop_event = Event()
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
        
    def start(self):
        """Start monitoring."""
        if self.monitor_thread is not None:
            return
            
        self.stop_event.clear()
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring."""
        if self.monitor_thread is None:
            return
            
        self.stop_event.set()
        self.monitor_thread.join()
        self.monitor_thread = None
        
    def _monitor_loop(self):
        """Monitoring loop."""
        while not self.stop_event.is_set():
            try:
                stats = self._get_stats()
                self.stats_queue.put(stats)
            except Exception as e:
                self.logger.error(f"Error in hardware monitor: {e}")
                
            time.sleep(self.interval)
            
    def _get_stats(self) -> HardwareStats:
        """Get current hardware statistics."""
        # Get GPU stats
        gpu_stats = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        
        if gpu_stats:
            gpu_util = gpu_stats.load * 100
            gpu_mem_used = gpu_stats.memoryUsed
            gpu_mem_total = gpu_stats.memoryTotal
            temperature = gpu_stats.temperature
            power = gpu_stats.powerUsage if hasattr(gpu_stats, 'powerUsage') else None
        else:
            gpu_util = 0.0
            gpu_mem_used = 0.0
            gpu_mem_total = 0.0
            temperature = None
            power = None
            
        # Get CPU stats
        cpu_util = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        cpu_mem_used = mem.used / (1024 ** 3)  # GB
        cpu_mem_total = mem.total / (1024 ** 3)  # GB
        
        return HardwareStats(
            gpu_utilization=gpu_util,
            gpu_memory_used=gpu_mem_used,
            gpu_memory_total=gpu_mem_total,
            cpu_utilization=cpu_util,
            cpu_memory_used=cpu_mem_used,
            cpu_memory_total=cpu_mem_total,
            temperature=temperature,
            power_usage=power
        )
        
    def get_stats(self) -> Optional[HardwareStats]:
        """Get latest statistics."""
        try:
            return self.stats_queue.get_nowait()
        except queue.Empty:
            return None
            
    def get_average_stats(self, window: int = 10) -> Optional[HardwareStats]:
        """Get average statistics over a window."""
        stats_list = []
        while len(stats_list) < window:
            try:
                stats = self.stats_queue.get_nowait()
                stats_list.append(stats)
            except queue.Empty:
                break
                
        if not stats_list:
            return None
            
        # Calculate averages
        avg_stats = HardwareStats(
            gpu_utilization=np.mean([s.gpu_utilization for s in stats_list]),
            gpu_memory_used=np.mean([s.gpu_memory_used for s in stats_list]),
            gpu_memory_total=stats_list[0].gpu_memory_total,
            cpu_utilization=np.mean([s.cpu_utilization for s in stats_list]),
            cpu_memory_used=np.mean([s.cpu_memory_used for s in stats_list]),
            cpu_memory_total=stats_list[0].cpu_memory_total,
            temperature=np.mean([s.temperature for s in stats_list if s.temperature is not None]),
            power_usage=np.mean([s.power_usage for s in stats_list if s.power_usage is not None])
        )
        
        return avg_stats

class HardwareManager:
    """Manages hardware resources and distributed training."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hardware manager.
        
        Args:
            config: Hardware configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.monitor = HardwareMonitor()
        
        # Initialize distributed training state
        self.distributed = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.device = None
        
        # Set up hardware
        self._setup_hardware()
        
    def _setup_hardware(self):
        """Set up hardware environment."""
        if torch.cuda.is_available():
            # Set CUDA device
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                gpu_id = self._select_best_gpu()
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                
            # Initialize distributed training if enabled
            if self.config.get('distributed', False):
                self._setup_distributed()
            else:
                self.device = torch.device('cuda:0')
                
            # Set CUDA options
            torch.backends.cudnn.benchmark = self.config.get('cudnn_benchmark', True)
            torch.backends.cudnn.deterministic = self.config.get('cudnn_deterministic', False)
            
            # Start hardware monitoring
            self.monitor.start()
        else:
            self.device = torch.device('cpu')
            warnings.warn("CUDA is not available. Using CPU.")
            
    def _select_best_gpu(self) -> int:
        """Select best GPU based on memory and utilization."""
        gpus = GPUtil.getGPUs()
        if not gpus:
            return 0
            
        # Score GPUs based on memory and utilization
        scores = []
        for gpu in gpus:
            memory_score = (gpu.memoryTotal - gpu.memoryUsed) / gpu.memoryTotal
            util_score = 1 - gpu.load
            scores.append(memory_score * 0.7 + util_score * 0.3)
            
        return int(np.argmax(scores))
        
    def _setup_distributed(self):
        """Set up distributed training."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )
        
        self.distributed = True
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        # Set device
        torch.cuda.set_device(self.local_rank)
        
    def optimize_memory(self):
        """Optimize memory usage."""
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Clear CPU cache
        import gc
        gc.collect()
        
        # Optimize CUDA memory allocator
        if hasattr(torch.cuda, 'memory_summary'):
            torch.cuda.memory_summary(device=None, abbreviated=False)
            
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        stats = {}
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9,
                'gpu_memory_cached': torch.cuda.memory_reserved() / 1e9,
                'gpu_max_memory': torch.cuda.max_memory_allocated() / 1e9
            })
            
        mem = psutil.virtual_memory()
        stats.update({
            'cpu_memory_used': mem.used / 1e9,
            'cpu_memory_available': mem.available / 1e9,
            'cpu_memory_percent': mem.percent
        })
        
        return stats
        
    def get_optimal_batch_size(self, 
                             model: torch.nn.Module,
                             input_size: Tuple[int, ...],
                             min_batch: int = 1,
                             max_batch: int = 512,
                             step_size: int = 1) -> int:
        """Find optimal batch size that fits in memory."""
        if not torch.cuda.is_available():
            return min_batch
            
        # Start with small batch
        batch_size = min_batch
        
        while batch_size <= max_batch:
            try:
                # Try batch size
                x = torch.randn(batch_size, *input_size).to(self.device)
                model(x)
                torch.cuda.empty_cache()
                
                # If successful, try larger batch
                batch_size += step_size
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Return last successful batch size
                    return max(min_batch, batch_size - step_size)
                else:
                    raise
                    
        return max_batch
        
    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        if torch.cuda.is_available() and self.config.get('mixed_precision', False):
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
            
    def synchronize(self):
        """Synchronize processes in distributed training."""
        if self.distributed:
            torch.cuda.synchronize()
            dist.barrier()
            
    def cleanup(self):
        """Clean up hardware manager."""
        self.monitor.stop()
        
        if self.distributed:
            dist.destroy_process_group()
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        
    def __str__(self) -> str:
        """String representation."""
        lines = ["Hardware Configuration:"]
        
        if torch.cuda.is_available():
            lines.extend([
                f"GPU: {torch.cuda.get_device_name(0)}",
                f"CUDA Version: {torch.version.cuda}",
                f"Distributed: {self.distributed}",
                f"World Size: {self.world_size}",
                f"Rank: {self.rank}",
                f"Local Rank: {self.local_rank}"
            ])
        else:
            lines.append("Device: CPU")
            
        return "\n".join(lines)