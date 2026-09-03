"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary."""
    return {
        'model_type': 'CNN_ResNet_Denoiser',
        'precision': 'INT8',
        'max_inference_latency_us': 120,
        'acceleration_target': 'eASIC_TensorCore',
        'model_path': 'models/denoiser_v120.engine'
    }


@pytest.fixture
def mock_srs_data():
    """Generate mock SRS channel data for testing."""
    # Shape: (Samples, Channels, Subcarriers, Symbols)
    return np.random.randn(10, 2, 72, 14).astype(np.complex64)


@pytest.fixture
def mock_iq_samples():
    """Generate mock IQ samples for testing."""
    # Complex-valued IQ samples
    real = np.random.randn(100, 10)
    imag = np.random.randn(100, 10)
    return real + 1j * imag


@pytest.fixture
def router_config():
    """Provide Router Agent configuration."""
    return {
        'snr_threshold_db': 5.0,
        'doppler_max_hz': 300.0,
        'execution_target': 'FPGA_Inline'
    }


@pytest.fixture
def drift_safety_config():
    """Provide Drift & Safety Agent configuration."""
    return {
        'max_evm_threshold': 0.18,
        'consecutive_failures_allowed': 3,
        'fallback_target': 'LMMSE_Hardware_Block',
        'telemetry_interval_ms': 10
    }
