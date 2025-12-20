"""GPU acceleration utilities for local backends."""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import GPU libraries
HAS_TORCH = False
HAS_CUDA = False
HAS_MPS = False
HAS_ROCM = False

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
    if hasattr(torch.backends, 'mps'):
        HAS_MPS = torch.backends.mps.is_available()
    # ROCm support (AMD GPUs)
    try:
        HAS_ROCM = torch.version.hip is not None
    except:
        HAS_ROCM = False
except ImportError:
    pass


def detect_gpu() -> Dict[str, Any]:
    """
    Detect available GPU devices.
    
    Returns:
        Dictionary with GPU information:
        {
            'available': bool,
            'device_type': str,  # 'cuda', 'mps', 'rocm', 'cpu'
            'device_name': str,
            'device_count': int,
            'memory_gb': Optional[float]
        }
    """
    info = {
        'available': False,
        'device_type': 'cpu',
        'device_name': 'CPU',
        'device_count': 0,
        'memory_gb': None
    }
    
    if not HAS_TORCH:
        return info
    
    # Check CUDA (NVIDIA)
    if HAS_CUDA:
        info['available'] = True
        info['device_type'] = 'cuda'
        info['device_count'] = torch.cuda.device_count()
        if info['device_count'] > 0:
            info['device_name'] = torch.cuda.get_device_name(0)
            try:
                # Get memory in GB
                memory_bytes = torch.cuda.get_device_properties(0).total_memory
                info['memory_gb'] = memory_bytes / (1024 ** 3)
            except:
                pass
        return info
    
    # Check MPS (Apple Silicon)
    if HAS_MPS:
        info['available'] = True
        info['device_type'] = 'mps'
        info['device_name'] = 'Apple Silicon GPU'
        info['device_count'] = 1
        # MPS doesn't expose memory info easily
        return info
    
    # Check ROCm (AMD)
    if HAS_ROCM:
        info['available'] = True
        info['device_type'] = 'rocm'
        info['device_name'] = 'AMD GPU'
        info['device_count'] = 1
        return info
    
    return info


def get_optimal_device(device_preference: Optional[str] = None) -> str:
    """
    Get optimal device for computation.
    
    Args:
        device_preference: Preferred device ('cuda', 'mps', 'rocm', 'cpu'). Auto-detects if None.
    
    Returns:
        Device string ('cuda', 'mps', 'rocm', or 'cpu')
    """
    gpu_info = detect_gpu()
    
    # If preference specified and available, use it
    if device_preference:
        if device_preference == 'cuda' and HAS_CUDA:
            return 'cuda'
        elif device_preference == 'mps' and HAS_MPS:
            return 'mps'
        elif device_preference == 'rocm' and HAS_ROCM:
            return 'rocm'
        elif device_preference == 'cpu':
            return 'cpu'
        else:
            logger.warning(f"Preferred device '{device_preference}' not available, auto-detecting")
    
    # Auto-detect best available
    if gpu_info['available']:
        return gpu_info['device_type']
    
    return 'cpu'


def enable_gpu_for_backend(backend: str, device_preference: Optional[str] = None) -> Optional[str]:
    """
    Enable GPU acceleration for a specific backend.
    
    Args:
        backend: Backend name ('ollama', 'huggingface', etc.)
        device_preference: Preferred device. Auto-detects if None.
    
    Returns:
        Device string if GPU is available and backend supports it, None otherwise
    """
    gpu_backends = {'ollama', 'huggingface', 'local'}
    
    if backend.lower() not in gpu_backends:
        return None
    
    device = get_optimal_device(device_preference)
    
    if device == 'cpu':
        return None
    
    logger.info(f"GPU acceleration enabled for {backend} on {device}")
    return device


def configure_torch_device(device_preference: Optional[str] = None) -> Any:
    """
    Configure PyTorch device for optimal performance.
    
    Args:
        device_preference: Preferred device. Auto-detects if None.
    
    Returns:
        torch.device object
    """
    if not HAS_TORCH:
        return None
    
    device_str = get_optimal_device(device_preference)
    return torch.device(device_str)


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get GPU memory information.
    
    Returns:
        Dictionary with memory info:
        {
            'total_gb': float,
            'allocated_gb': float,
            'free_gb': float,
            'cached_gb': float
        }
    """
    if not HAS_TORCH or not HAS_CUDA:
        return {
            'total_gb': 0.0,
            'allocated_gb': 0.0,
            'free_gb': 0.0,
            'cached_gb': 0.0
        }
    
    try:
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        cached = torch.cuda.memory_reserved(0) / (1024 ** 3)
        free = total - cached
        
        return {
            'total_gb': total,
            'allocated_gb': allocated,
            'free_gb': free,
            'cached_gb': cached
        }
    except Exception as e:
        logger.warning(f"Could not get GPU memory info: {e}")
        return {
            'total_gb': 0.0,
            'allocated_gb': 0.0,
            'free_gb': 0.0,
            'cached_gb': 0.0
        }








