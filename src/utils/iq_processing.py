"""
IQ Processing Utilities: Real/Imaginary Channel Tensor Parsing

Transforms complex-valued IQ samples (I + jQ) into 2-channel tensors
(Real/Imaginary) suitable for neural network inference.
"""

import numpy as np
from typing import Tuple


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
    Compute instantaneous power from IQ tensor.
    
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
    
    Args:
        srs_grid: Input SRS grid
        window_type: 'hann', 'hamming', 'blackman'
        
    Returns:
        Windowed SRS grid
    """
    if window_type == 'hann':
        window = np.hanning(srs_grid.shape[-1])
    elif window_type == 'hamming':
        window = np.hamming(srs_grid.shape[-1])
    elif window_type == 'blackman':
        window = np.blackman(srs_grid.shape[-1])
    else:
        window = np.ones(srs_grid.shape[-1])
    
    return srs_grid * window
