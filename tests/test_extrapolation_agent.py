"""Unit tests for the Extrapolation Agent."""

import pytest
import numpy as np
from src.agents.extrapolator import ExtrapolationAgent


class TestExtrapolationAgent:
    """Test suite for ExtrapolationAgent class."""

    def test_initialization(self, mock_config):
        """Test ExtrapolationAgent initialization with config."""
        agent = ExtrapolationAgent(mock_config)
        assert agent.model_type == 'Sparse_Transformer_Encoder'
        assert agent.precision == 'FPGA_FIXED_16'
        assert agent.max_latency_us == 80
        assert agent.acceleration_target == 'SmartNIC_Core'

    def test_initialization_with_defaults(self):
        """Test ExtrapolationAgent initialization with empty config uses defaults."""
        agent = ExtrapolationAgent({})
        assert agent.model_type == 'Sparse_Transformer_Encoder'
        assert agent.precision == 'FPGA_FIXED_16'
        assert agent.max_latency_us == 80
        assert agent.acceleration_target == 'SmartNIC_Core'

    def test_extrapolate_channel_shape_preservation(self, mock_config, mock_srs_data):
        """Test that extrapolation preserves input shape."""
        agent = ExtrapolationAgent(mock_config)
        sparse_input = mock_srs_data[:, :, ::2, :]  # Reduce subcarriers
        output = agent.extrapolate_channel(sparse_input)
        assert output.shape == sparse_input.shape

    def test_extrapolate_channel_returns_ndarray(self, mock_config, mock_srs_data):
        """Test that extrapolation returns a numpy array."""
        agent = ExtrapolationAgent(mock_config)
        output = agent.extrapolate_channel(mock_srs_data)
        assert isinstance(output, np.ndarray)

    def test_extrapolate_sparse_grid(self, mock_config):
        """Test extrapolation with sparse grid input."""
        agent = ExtrapolationAgent(mock_config)
        # Simulate sparse grid with reduced pilot subcarriers
        sparse_grid = np.random.randn(8, 4, 36, 14).astype(np.complex64)
        output = agent.extrapolate_channel(sparse_grid)
        assert output.shape == sparse_grid.shape

    def test_get_latency_metrics(self, mock_config):
        """Test retrieval of latency metrics."""
        agent = ExtrapolationAgent(mock_config)
        metrics = agent.get_latency_metrics()
        
        assert isinstance(metrics, dict)
        assert 'model_type' in metrics
        assert 'precision' in metrics
        assert 'max_latency_us' in metrics
        assert 'acceleration_target' in metrics
        
        assert metrics['model_type'] == 'Sparse_Transformer_Encoder'
        assert metrics['precision'] == 'FPGA_FIXED_16'
        assert metrics['max_latency_us'] == 80
        assert metrics['acceleration_target'] == 'SmartNIC_Core'

    def test_latency_budget_compliance(self, mock_config):
        """Test that max latency is within O-RAN budget."""
        agent = ExtrapolationAgent(mock_config)
        assert agent.max_latency_us <= 80  # SmartNIC compliance

    def test_custom_model_type(self):
        """Test initialization with custom model type."""
        config = {'model_type': 'Vision_Transformer_v3'}
        agent = ExtrapolationAgent(config)
        assert agent.model_type == 'Vision_Transformer_v3'

    def test_precision_options(self):
        """Test different precision configurations."""
        for precision in ['FPGA_FIXED_16', 'INT8', 'FP32']:
            config = {'precision': precision}
            agent = ExtrapolationAgent(config)
            assert agent.precision == precision

    def test_custom_latency_budget(self):
        """Test custom latency budget configuration."""
        config = {'max_inference_latency_us': 100}
        agent = ExtrapolationAgent(config)
        assert agent.max_latency_us == 100

    def test_extrapolate_1d_array(self, mock_config):
        """Test extrapolation with 1D array."""
        agent = ExtrapolationAgent(mock_config)
        input_1d = np.random.randn(72).astype(np.complex64)
        output = agent.extrapolate_channel(input_1d)
        assert output.shape == input_1d.shape

    def test_extrapolate_high_dimensional_array(self, mock_config):
        """Test extrapolation with high-dimensional array."""
        agent = ExtrapolationAgent(mock_config)
        input_5d = np.random.randn(2, 2, 36, 7, 2).astype(np.complex64)
        output = agent.extrapolate_channel(input_5d)
        assert output.shape == input_5d.shape
