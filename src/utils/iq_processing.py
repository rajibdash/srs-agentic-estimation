"""
IQ Processing Utilities: Real/Imaginary Channel Tensor Parsing

Transforms complex-valued IQ samples (I + jQ) into 2-channel tensors
(Real/Imaginary) suitable for neural network inference.
"""

import numpy as np
from typing import Tuple, Optional
from functools import lru_cache


# Pre-computed windows cache
_WINDOW_CACHE = {}


def _get_cached_window(window_type: str, length: int) -> np.ndarray:
    """
    Get or create a pre-computed window function.
    
    Args:
        window_type: 'hann', 'hamming', 'blackman'
        length: Window length
        
    Returns:
        Cached window array
    """
    key = (window_type, length)
    if key not in _WINDOW_CACHE:
        if window_type == 'hann':
            _WINDOW_CACHE[key] = np.hanning(length)
        elif window_type == 'hamming':
            _WINDOW_CACHE[key] = np.hamming(length)
        elif window_type == 'blackman':
            _WINDOW_CACHE[key] = np.blackman(length)
        else:
            _WINDOW_CACHE[key] = np.ones(length)
    return _WINDOW_CACHE[key]


def parse_complex_iq_to_tensor(iq_samples: np.ndarray) -> np.ndarray:
    """
    Convert complex-valued IQ samples to real/imaginary 2-channel tensor.
    
    Args:
        iq_samples: Complex numpy array of shape (num_samples,) or (num_samples, num_features)
        
    Returns:
        Real/Imaginary tensor of shape (..., 2) with last dimension being [Real, Imag]
    """
    real_part = np.real(iq_samples)
    imag_part = np.imag(iq_samples)
    
    # Stack along last dimension
    return np.stack([real_part, imag_part], axis=-1).astype(np.float32)


def tensor_to_complex_iq(tensor: np.ndarray) -> np.ndarray:
    """
    Convert real/imaginary 2-channel tensor back to complex IQ samples.
    
    Args:
        tensor: Real/Imaginary tensor with shape (..., 2)
        
    Returns:
        Complex numpy array
    """
    real_part = tensor[..., 0]
    imag_part = tensor[..., 1]
    return real_part + 1j * imag_part


class IncrementalStatistics:
    """
    Efficient incremental statistics computation using Welford's algorithm.
    Avoids recomputing mean/std from scratch on every call.
    """
    
    def __init__(self, max_samples: int = 10000):
        """
        Initialize incremental statistics tracker.
        
        Args:
            max_samples: Maximum samples to track before rotation
        """
        self.max_samples = max_samples
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Welford's M2 for variance
    
    def update(self, value: float) -> None:
        """
        Update statistics with new value using Welford's algorithm.
        
        Args:
            value: New data point
        """
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
    
    def get_mean(self) -> float:
        """Get current mean."""
        return self.mean
    
    def get_std(self) -> float:
        """Get current standard deviation."""
        if self.count < 2:
            return 0.0
        return np.sqrt(self.M2 / (self.count - 1))
    
    def reset(self) -> None:
        """Reset statistics."""
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0


def normalize_iq_tensor(tensor: np.ndarray, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """
    Normalize IQ tensor to zero mean and unit variance.
    
    Args:
        tensor: Input tensor
        mean: Target mean (default 0.0)
        std: Target standard deviation (default 1.0)
        
    Returns:
        Normalized tensor
    """
    tensor_mean = np.mean(tensor)
    tensor_std = np.std(tensor)
    
    if tensor_std == 0:
        return tensor
    
    normalized = (tensor - tensor_mean) / tensor_std
    return normalized * std + mean


def compute_power(iq_tensor: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous power from IQ tensor using vectorized operations.
    
    Power = Real^2 + Imaginary^2
    
    Args:
        iq_tensor: Real/Imaginary tensor
        
    Returns:
        Power array
    """
    real_squared = iq_tensor[..., 0] ** 2
    imag_squared = iq_tensor[..., 1] ** 2
    return real_squared + imag_squared


def apply_windowing(srs_grid: np.ndarray, window_type: str = 'hann') -> np.ndarray:
    """
    Apply windowing function to SRS grid to reduce spectral leakage.
    Uses cached window functions to avoid recomputation.
    
    Args:
        srs_grid: Input SRS grid
        window_type: 'hann', 'hamming', 'blackman'
        
    Returns:
        Windowed SRS grid
    """
    window = _get_cached_window(window_type, srs_grid.shape[-1])
    return srs_grid * window


def clear_window_cache() -> None:
    """
    Clear the window function cache (useful for testing/memory management).
    """
    global _WINDOW_CACHE
    _WINDOW_CACHE.clear()
