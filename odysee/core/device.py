import numpy as np
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

class Device:
    """Device abstraction for CPU/GPU computation"""
    def __init__(self, device_type: str = "cpu"):
        if device_type == "cuda" and not CUDA_AVAILABLE:
            raise RuntimeError("CUDA device requested but cupy is not installed")
        self.type = device_type
        self.xp = cp if device_type == "cuda" else np
        
    @property
    def is_cuda(self) -> bool:
        return self.type == "cuda"
    
    def __str__(self) -> str:
        return self.type
    
    def transfer_array(self, array):
        """Transfer array to this device"""
        if self.is_cuda:
            if isinstance(array, np.ndarray):
                return cp.array(array)
        else:
            if hasattr(array, "get"):
                return array.get()
        return array

# Global device registry
cpu = Device("cpu")
cuda = Device("cuda") if CUDA_AVAILABLE else None

def get_default_device():
    """Get default device (CUDA if available, else CPU)"""
    return cuda if CUDA_AVAILABLE else cpu
