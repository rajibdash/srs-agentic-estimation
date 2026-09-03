"""Unit tests for the MLOps orchestrator."""

import pytest
from pipelines.mlops_orchestrator import SRSMLOpsPipeline


class TestSRSMLOpsPipeline:
    """Test suite for SRSMLOpsPipeline class."""

    def test_initialization(self):
        """Test MLOps pipeline initialization."""
        agent_name = 'Denoising_Agent'
        pipeline = SRSMLOpsPipeline(agent_name)
        
        assert pipeline.agent_name == agent_name
        assert pipeline.feature_store == 'mock_tensor_stream://feature-store.local'
        assert pipeline.registry_url == 'mock_registry://gnodeb-models.internal'

    def test_initialization_different_agents(self):
        """Test initialization with different agent names."""
        for agent_name in ['Denoising_Agent', 'Extrapolation_Agent']:
            pipeline = SRSMLOpsPipeline(agent_name)
            assert pipeline.agent_name == agent_name

    def test_fetch_drifted_telemetry(self):
        """Test fetching drifted telemetry data."""
        pipeline = SRSMLOpsPipeline('Denoising_Agent')
        data_ref = pipeline.fetch_drifted_telemetry()
        
        assert isinstance(data_ref, str)
        assert data_ref == 'raw_tensors_v12'

    def test_execute_retraining(self):
        """Test retraining execution."""
        pipeline = SRSMLOpsPipeline('Denoising_Agent')
        data_ref = 'raw_tensors_v12'
        model_path = pipeline.execute_retraining(data_ref)
        
        assert isinstance(model_path, str)
        assert model_path == 'refined_srs_model.onnx'

    def test_convert_and_quantize(self):
        """Test model quantization."""
        pipeline = SRSMLOpsPipeline('Denoising_Agent')
        model_path = 'refined_srs_model.onnx'
        quantized_path = pipeline.convert_and_quantize(model_path)
        
        assert isinstance(quantized_path, str)
        assert quantized_path == 'refined_srs_model_int8.bin'

    def test_deploy_to_gnodeb_du(self):
        """Test deployment to gNodeB Distributed Units."""
        pipeline = SRSMLOpsPipeline('Denoising_Agent')
        deployment_package = 'refined_srs_model_int8.bin'
        success = pipeline.deploy_to_gnodeb_du(deployment_package)
        
        assert isinstance(success, bool)
        assert success is True

    def test_run_full_pipeline_denoising_agent(self):
        """Test complete pipeline for Denoising Agent."""
        pipeline = SRSMLOpsPipeline(agent_name='Denoising_Agent')
        success = pipeline.run_full_pipeline()
        
        assert isinstance(success, bool)
        assert success is True

    def test_run_full_pipeline_extrapolation_agent(self):
        """Test complete pipeline for Extrapolation Agent."""
        pipeline = SRSMLOpsPipeline(agent_name='Extrapolation_Agent')
        success = pipeline.run_full_pipeline()
        
        assert isinstance(success, bool)
        assert success is True

    def test_pipeline_returns_success_on_completion(self):
        """Test that pipeline returns success status."""
        pipeline = SRSMLOpsPipeline('Denoising_Agent')
        success = pipeline.run_full_pipeline()
        assert success is True

    def test_multiple_pipeline_runs(self):
        """Test running multiple pipeline instances sequentially."""
        pipeline1 = SRSMLOpsPipeline('Denoising_Agent')
        result1 = pipeline1.run_full_pipeline()
        
        pipeline2 = SRSMLOpsPipeline('Extrapolation_Agent')
        result2 = pipeline2.run_full_pipeline()
        
        assert result1 is True
        assert result2 is True
