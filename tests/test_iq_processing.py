"""Unit tests for IQ processing utilities."""

import pytest
import numpy as np
from src.utils.iq_processing import (
    parse_complex_iq_to_tensor,
    tensor_to_complex_iq,
    normalize_iq_tensor,
    compute_power,
    apply_windowing
)


class TestIQProcessing:
    """Test suite for IQ processing utility functions."""

    def test_parse_complex_iq_to_tensor_1d(self):
        """Test conversion of 1D complex IQ samples to tensor."""
        iq_samples = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
        result = parse_complex_iq_to_tensor(iq_samples)
        
        assert result.shape == (3, 2)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result[:, 0], [1, 3, 5])
        np.testing.assert_array_almost_equal(result[:, 1], [2, 4, 6])

    def test_parse_complex_iq_to_tensor_2d(self):
        """Test conversion of 2D complex IQ samples to tensor."""
        iq_samples = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
        result = parse_complex_iq_to_tensor(iq_samples)
        
        assert result.shape == (2, 2, 2)
        assert result.dtype == np.float32

    def test_parse_complex_iq_to_tensor_multidimensional(self, mock_iq_samples):
        """Test conversion with arbitrary dimensional complex arrays."""
        result = parse_complex_iq_to_tensor(mock_iq_samples)
        
        original_shape = mock_iq_samples.shape
        expected_shape = original_shape + (2,)
        assert result.shape == expected_shape
        assert result.dtype == np.float32

    def test_tensor_to_complex_iq_1d(self):
        """Test conversion from 1D real/imaginary tensor back to complex."""
        tensor = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        result = tensor_to_complex_iq(tensor)
        
        expected = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
        np.testing.assert_array_almost_equal(result, expected)

    def test_tensor_to_complex_iq_2d(self):
        """Test conversion from 2D real/imaginary tensor back to complex."""
        tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
        result = tensor_to_complex_iq(tensor)
        
        assert result.shape == (2, 2)
        assert np.iscomplexobj(result)

    def test_roundtrip_conversion(self, mock_iq_samples):
        """Test roundtrip conversion: complex -> tensor -> complex."""
        tensor = parse_complex_iq_to_tensor(mock_iq_samples)
        recovered = tensor_to_complex_iq(tensor)
        
        np.testing.assert_array_almost_equal(recovered, mock_iq_samples, decimal=5)

    def test_normalize_iq_tensor_default(self):
        """Test normalization with default mean=0, std=1."""
        tensor = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = normalize_iq_tensor(tensor)
        
        assert abs(np.mean(result)) < 1e-6
        assert abs(np.std(result) - 1.0) < 1e-6

    def test_normalize_iq_tensor_custom_mean_std(self):
        """Test normalization with custom mean and std."""
        tensor = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = normalize_iq_tensor(tensor, mean=10.0, std=2.0)
        
        assert abs(np.mean(result) - 10.0) < 1e-6
        assert abs(np.std(result) - 2.0) < 1e-6

    def test_normalize_iq_tensor_zero_std(self):
        """Test normalization with zero standard deviation."""
        tensor = np.array([5, 5, 5, 5], dtype=np.float32)
        result = normalize_iq_tensor(tensor)
        
        # Should return unchanged when std is 0
        np.testing.assert_array_equal(result, tensor)

    def test_normalize_iq_tensor_2d(self):
        """Test normalization with 2D tensor."""
        tensor = np.random.randn(10, 10).astype(np.float32)
        result = normalize_iq_tensor(tensor)
        
        assert abs(np.mean(result)) < 1e-6
        assert abs(np.std(result) - 1.0) < 1e-6

    def test_compute_power_1d(self):
        """Test power computation with 1D tensor."""
        # Real/imaginary tensor: [[3, 4]] -> power = 3^2 + 4^2 = 25
        tensor = np.array([[3, 4]], dtype=np.float32)
        result = compute_power(tensor)
        
        np.testing.assert_array_almost_equal(result, [25.0])

    def test_compute_power_2d(self):
        """Test power computation with 2D tensor."""
        tensor = np.array([[[1, 0], [0, 1]], [[3, 4], [0, 0]]], dtype=np.float32)
        result = compute_power(tensor)
        
        expected = np.array([[1, 1], [25, 0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_compute_power_shape(self):
        """Test that power computation reduces last dimension."""
        tensor = np.random.randn(10, 5, 2).astype(np.float32)
        result = compute_power(tensor)
        
        assert result.shape == (10, 5)

    def test_compute_power_always_positive(self):
        """Test that computed power is always non-negative."""
        tensor = np.random.randn(100, 50, 2).astype(np.float32)
        result = compute_power(tensor)
        
        assert np.all(result >= 0)

    def test_apply_windowing_hann(self):
        """Test Hann windowing application."""
        srs_grid = np.ones((5, 10), dtype=np.float32)
        result = apply_windowing(srs_grid, window_type='hann')
        
        assert result.shape == srs_grid.shape
        # Hann window should taper edges
        assert result[0, 0] < result[0, 5]  # Edge < center

    def test_apply_windowing_hamming(self):
        """Test Hamming windowing application."""
        srs_grid = np.ones((5, 10), dtype=np.float32)
        result = apply_windowing(srs_grid, window_type='hamming')
        
        assert result.shape == srs_grid.shape
        # Hamming window should taper edges
        assert result[0, 0] < result[0, 5]

    def test_apply_windowing_blackman(self):
        """Test Blackman windowing application."""
        srs_grid = np.ones((5, 10), dtype=np.float32)
        result = apply_windowing(srs_grid, window_type='blackman')
        
        assert result.shape == srs_grid.shape
        # Blackman window should taper edges
        assert result[0, 0] < result[0, 5]

    def test_apply_windowing_unknown_type(self):
        """Test windowing with unknown type defaults to rectangular."""
        srs_grid = np.ones((5, 10), dtype=np.float32)
        result = apply_windowing(srs_grid, window_type='unknown')
        
        # Should apply rectangular window (all ones)
        np.testing.assert_array_equal(result, srs_grid)

    def test_apply_windowing_multidimensional(self):
        """Test windowing with multi-dimensional input."""
        srs_grid = np.random.randn(3, 4, 10).astype(np.float32)
        result = apply_windowing(srs_grid, window_type='hann')
        
        assert result.shape == srs_grid.shape
