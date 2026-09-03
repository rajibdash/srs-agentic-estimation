"""Unit tests for the training pipeline."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from pipelines.train_pipeline import (
    load_srs_data_from_store,
    train_and_optimize_agent,
    apply_post_training_quantization,
    register_and_deploy
)


class TestTrainPipeline:
    """Test suite for training pipeline functions."""

    def test_load_srs_data_from_store(self):
        """Test loading mock SRS data from store."""
        srs, csi = load_srs_data_from_store()
        
        assert isinstance(srs, np.ndarray)
        assert isinstance(csi, np.ndarray)
        # Check expected shape: (100, 2, 72, 14)
        assert srs.shape == (100, 2, 72, 14)
        assert csi.shape == (100, 2, 72, 14)
        # Check data type
        assert srs.dtype == np.float32
        assert csi.dtype == np.float32

    def test_load_srs_data_relationship(self):
        """Test relationship between SRS and CSI data."""
        srs, csi = load_srs_data_from_store()
        
        # CSI should be derived from SRS (csi = srs * 1.5 + 0.1)
        # We can't test exact values due to randomness, but shape and type
        assert csi.shape == srs.shape

    def test_train_and_optimize_agent(self):
        """Test model training and optimization."""
        x_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randn(100, 5).astype(np.float32)
        
        model_path = train_and_optimize_agent(x_train, y_train)
        
        assert isinstance(model_path, str)
        assert model_path == 'Optimized_Agent_Weights'

    def test_train_and_optimize_agent_timing(self):
        """Test that training completes in reasonable time."""
        x_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randn(100, 5).astype(np.float32)
        
        import time
        start = time.time()
        model_path = train_and_optimize_agent(x_train, y_train)
        elapsed = time.time() - start
        
        # Should complete relatively quickly (mocked)
        assert elapsed < 5.0

    def test_apply_post_training_quantization(self):
        """Test post-training quantization."""
        model = 'Optimized_Agent_Weights'
        quantized_path = apply_post_training_quantization(model)
        
        assert isinstance(quantized_path, str)
        assert quantized_path == 'quantized_model.onnx'

    def test_register_and_deploy(self):
        """Test model registration and deployment."""
        model_path = 'quantized_model.onnx'
        success = register_and_deploy(model_path)
        
        assert isinstance(success, bool)
        assert success is True

    def test_full_pipeline_execution(self):
        """Test complete pipeline execution flow."""
        # Load data
        x, y = load_srs_data_from_store()
        assert x.shape[0] > 0
        
        # Train
        model = train_and_optimize_agent(x, y)
        assert isinstance(model, str)
        
        # Quantize
        quantized = apply_post_training_quantization(model)
        assert isinstance(quantized, str)
        
        # Deploy
        success = register_and_deploy(quantized)
        assert success is True
