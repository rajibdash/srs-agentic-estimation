"""Unit tests for the Denoising Agent."""

import pytest
import numpy as np
from src.agents.denoiser import DenoisingAgent


class TestDenoisingAgent:
    """Test suite for DenoisingAgent class."""

    def test_initialization(self, mock_config):
        """Test DenoisingAgent initialization with config."""
        agent = DenoisingAgent(mock_config)
        assert agent.model_type == 'CNN_ResNet_Denoiser'
        assert agent.precision == 'INT8'
        assert agent.max_latency_us == 120
        assert agent.acceleration_target == 'eASIC_TensorCore'
        assert agent.model_path == 'models/denoiser_v120.engine'

    def test_initialization_with_defaults(self):
        """Test DenoisingAgent initialization with empty config uses defaults."""
        agent = DenoisingAgent({})
        assert agent.model_type == 'CNN_ResNet_Denoiser'
        assert agent.precision == 'INT8'
        assert agent.max_latency_us == 120
        assert agent.acceleration_target == 'eASIC_TensorCore'

    def test_denoise_srs_matrix_shape(self, mock_config, mock_srs_data):
        """Test that denoising preserves input shape."""
        agent = DenoisingAgent(mock_config)
        input_shape = mock_srs_data.shape
        output = agent.denoise_srs_matrix(mock_srs_data)
        assert output.shape == input_shape

    def test_denoise_srs_matrix_returns_ndarray(self, mock_config, mock_srs_data):
        """Test that denoising returns a numpy array."""
        agent = DenoisingAgent(mock_config)
        output = agent.denoise_srs_matrix(mock_srs_data)
        assert isinstance(output, np.ndarray)

    def test_denoise_srs_matrix_input_output_consistency(self, mock_config):
        """Test denoising with simple input for consistency."""
        agent = DenoisingAgent(mock_config)
        simple_input = np.ones((4, 4), dtype=np.complex64)
        output = agent.denoise_srs_matrix(simple_input)
        assert output.shape == simple_input.shape

    def test_get_latency_metrics(self, mock_config):
        """Test retrieval of latency metrics."""
        agent = DenoisingAgent(mock_config)
        metrics = agent.get_latency_metrics()
        
        assert isinstance(metrics, dict)
        assert 'model_type' in metrics
        assert 'precision' in metrics
        assert 'max_latency_us' in metrics
        assert 'acceleration_target' in metrics
        
        assert metrics['model_type'] == 'CNN_ResNet_Denoiser'
        assert metrics['precision'] == 'INT8'
        assert metrics['max_latency_us'] == 120
        assert metrics['acceleration_target'] == 'eASIC_TensorCore'

    def test_latency_budget_compliance(self, mock_config):
        """Test that max latency is within budget."""
        agent = DenoisingAgent(mock_config)
        assert agent.max_latency_us <= 120  # O-RAN compliance

    def test_custom_model_type(self):
        """Test initialization with custom model type."""
        config = {'model_type': 'CustomDenoiser_v2'}
        agent = DenoisingAgent(config)
        assert agent.model_type == 'CustomDenoiser_v2'

    def test_precision_options(self):
        """Test different precision configurations."""
        for precision in ['INT8', 'FP16', 'FP32']:
            config = {'precision': precision}
            agent = DenoisingAgent(config)
            assert agent.precision == precision

    def test_custom_latency_budget(self):
        """Test custom latency budget configuration."""
        config = {'max_inference_latency_us': 100}
        agent = DenoisingAgent(config)
        assert agent.max_latency_us == 100

    def test_denoise_large_matrix(self, mock_config):
        """Test denoising with large matrix."""
        agent = DenoisingAgent(mock_config)
        large_input = np.random.randn(1000, 1000).astype(np.complex64)
        output = agent.denoise_srs_matrix(large_input)
        assert output.shape == large_input.shape

    def test_denoise_empty_matrix(self, mock_config):
        """Test denoising with empty matrix."""
        agent = DenoisingAgent(mock_config)
        empty_input = np.array([], dtype=np.complex64).reshape(0, 0)
        output = agent.denoise_srs_matrix(empty_input)
        assert output.shape == empty_input.shape
