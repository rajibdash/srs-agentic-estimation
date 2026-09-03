"""Unit tests for the Drift & Safety Agent."""

import pytest
import time
from src.agents.drift_safety import DriftSafetyAgent


class TestDriftSafetyAgent:
    """Test suite for DriftSafetyAgent class."""

    def test_initialization(self, drift_safety_config):
        """Test DriftSafetyAgent initialization with config."""
        agent = DriftSafetyAgent(drift_safety_config)
        assert agent.max_evm_threshold == 0.18
        assert agent.consecutive_failures_allowed == 3
        assert agent.fallback_target == 'LMMSE_Hardware_Block'
        assert agent.telemetry_interval_ms == 10
        assert agent.consecutive_failures == 0

    def test_initialization_with_defaults(self):
        """Test DriftSafetyAgent initialization with empty config uses defaults."""
        agent = DriftSafetyAgent({})
        assert agent.max_evm_threshold == 0.18
        assert agent.consecutive_failures_allowed == 3
        assert agent.fallback_target == 'LMMSE_Hardware_Block'
        assert agent.telemetry_interval_ms == 10

    def test_evaluate_model_performance_healthy(self, drift_safety_config):
        """Test model performance evaluation when healthy."""
        agent = DriftSafetyAgent(drift_safety_config)
        is_healthy, message = agent.evaluate_model_performance(bler=0.05, evm=0.10)
        assert is_healthy is True
        assert 'acceptable bounds' in message

    def test_evaluate_model_performance_single_failure(self, drift_safety_config):
        """Test model performance with single EVM threshold violation."""
        agent = DriftSafetyAgent(drift_safety_config)
        is_healthy, message = agent.evaluate_model_performance(bler=0.05, evm=0.20)
        # Single failure should not trigger drift yet
        assert is_healthy is True
        assert agent.consecutive_failures == 1

    def test_evaluate_model_performance_drift_detected(self, drift_safety_config):
        """Test drift detection after consecutive failures."""
        agent = DriftSafetyAgent(drift_safety_config)
        # First failure
        agent.evaluate_model_performance(bler=0.05, evm=0.20)
        # Second failure
        agent.evaluate_model_performance(bler=0.05, evm=0.20)
        # Third failure (should trigger drift)
        is_healthy, message = agent.evaluate_model_performance(bler=0.05, evm=0.20)
        assert is_healthy is False
        assert 'DRIFT DETECTED' in message
        assert '0.200' in message

    def test_consecutive_failures_reset_on_success(self, drift_safety_config):
        """Test that consecutive failure counter resets on successful evaluation."""
        agent = DriftSafetyAgent(drift_safety_config)
        # First failure
        agent.evaluate_model_performance(bler=0.05, evm=0.20)
        assert agent.consecutive_failures == 1
        # Success (reset counter)
        agent.evaluate_model_performance(bler=0.05, evm=0.10)
        assert agent.consecutive_failures == 0

    def test_trigger_retraining(self, drift_safety_config):
        """Test retraining trigger returns proper metadata."""
        agent = DriftSafetyAgent(drift_safety_config)
        # Add some telemetry data
        agent.evaluate_model_performance(bler=0.05, evm=0.10)
        agent.evaluate_model_performance(bler=0.06, evm=0.11)
        
        retraining_meta = agent.trigger_retraining()
        
        assert retraining_meta['status'] == 'RETRAINING_TRIGGERED'
        assert 'bler_samples' in retraining_meta
        assert 'evm_samples' in retraining_meta
        assert 'timestamp' in retraining_meta
        assert len(retraining_meta['bler_samples']) == 2
        assert len(retraining_meta['evm_samples']) == 2

    def test_trigger_fallback(self, drift_safety_config):
        """Test fallback trigger returns target algorithm."""
        agent = DriftSafetyAgent(drift_safety_config)
        fallback_target = agent.trigger_fallback()
        assert fallback_target == 'LMMSE_Hardware_Block'

    def test_get_telemetry(self, drift_safety_config):
        """Test telemetry snapshot generation."""
        agent = DriftSafetyAgent(drift_safety_config)
        agent.evaluate_model_performance(bler=0.05, evm=0.10)
        agent.evaluate_model_performance(bler=0.06, evm=0.12)
        
        telemetry = agent.get_telemetry()
        
        assert 'timestamp' in telemetry
        assert 'consecutive_failures' in telemetry
        assert 'bler_average' in telemetry
        assert 'evm_average' in telemetry
        assert 'bler_history_length' in telemetry
        assert 'evm_history_length' in telemetry
        
        assert telemetry['bler_history_length'] == 2
        assert telemetry['evm_history_length'] == 2
        assert abs(telemetry['bler_average'] - 0.055) < 0.01
        assert abs(telemetry['evm_average'] - 0.11) < 0.01

    def test_telemetry_empty_history(self, drift_safety_config):
        """Test telemetry with empty history."""
        agent = DriftSafetyAgent(drift_safety_config)
        telemetry = agent.get_telemetry()
        
        assert telemetry['bler_average'] == 0.0
        assert telemetry['evm_average'] == 0.0
        assert telemetry['bler_history_length'] == 0
        assert telemetry['evm_history_length'] == 0

    def test_history_max_length(self, drift_safety_config):
        """Test that history buffers respect max length."""
        agent = DriftSafetyAgent(drift_safety_config)
        # Add more than 1000 samples
        for i in range(1100):
            agent.evaluate_model_performance(bler=0.05, evm=0.10)
        
        # History should be capped at 1000
        assert len(agent.bler_history) == 1000
        assert len(agent.evm_history) == 1000

    def test_custom_evm_threshold(self):
        """Test custom EVM threshold configuration."""
        config = {'max_evm_threshold': 0.25}
        agent = DriftSafetyAgent(config)
        is_healthy, _ = agent.evaluate_model_performance(bler=0.05, evm=0.24)
        assert is_healthy is True

    def test_custom_failure_threshold(self):
        """Test custom consecutive failure threshold."""
        config = {'consecutive_failures_allowed': 5}
        agent = DriftSafetyAgent(config)
        # 4 failures should not trigger drift
        for _ in range(4):
            is_healthy, _ = agent.evaluate_model_performance(bler=0.05, evm=0.20)
            assert is_healthy is True
        # 5th failure should trigger drift
        is_healthy, _ = agent.evaluate_model_performance(bler=0.05, evm=0.20)
        assert is_healthy is False

    def test_zero_bler_evm(self, drift_safety_config):
        """Test with zero BLER and EVM values."""
        agent = DriftSafetyAgent(drift_safety_config)
        is_healthy, message = agent.evaluate_model_performance(bler=0.0, evm=0.0)
        assert is_healthy is True
