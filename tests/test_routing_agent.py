"""Unit tests for the Routing Agent."""

import pytest
from src.agents.router import RoutingAgent


class TestRoutingAgent:
    """Test suite for RoutingAgent class."""

    def test_initialization(self, router_config):
        """Test RoutingAgent initialization with config."""
        agent = RoutingAgent(router_config)
        assert agent.snr_threshold_db == 5.0
        assert agent.doppler_max_hz == 300.0
        assert agent.execution_target == 'FPGA_Inline'

    def test_initialization_with_defaults(self):
        """Test RoutingAgent initialization with empty config uses defaults."""
        agent = RoutingAgent({})
        assert agent.snr_threshold_db == 12.5
        assert agent.doppler_max_hz == 300
        assert agent.execution_target == 'FPGA_Inline'

    def test_evaluate_channel_state_poor_snr(self, router_config):
        """Test channel evaluation with poor SNR."""
        agent = RoutingAgent(router_config)
        result = agent.evaluate_channel_state(rssi=-80, snr=3.0, doppler=100.0)
        assert result == 'POOR_CHANNEL'

    def test_evaluate_channel_state_good_snr(self, router_config):
        """Test channel evaluation with good SNR."""
        agent = RoutingAgent(router_config)
        result = agent.evaluate_channel_state(rssi=-60, snr=15.0, doppler=100.0)
        assert result == 'GOOD_CHANNEL'

    def test_evaluate_channel_state_high_doppler(self, router_config):
        """Test channel evaluation with high Doppler spread."""
        agent = RoutingAgent(router_config)
        result = agent.evaluate_channel_state(rssi=-60, snr=15.0, doppler=400.0)
        assert result == 'POOR_CHANNEL'

    def test_evaluate_channel_state_boundary_snr(self, router_config):
        """Test channel evaluation at SNR threshold boundary."""
        agent = RoutingAgent(router_config)
        # Exactly at threshold should be GOOD_CHANNEL
        result = agent.evaluate_channel_state(rssi=-60, snr=5.0, doppler=100.0)
        assert result == 'GOOD_CHANNEL'

    def test_evaluate_channel_state_just_below_threshold(self, router_config):
        """Test channel evaluation just below SNR threshold."""
        agent = RoutingAgent(router_config)
        result = agent.evaluate_channel_state(rssi=-60, snr=4.99, doppler=100.0)
        assert result == 'POOR_CHANNEL'

    def test_route_workload_poor_channel(self, router_config):
        """Test workload routing for poor channel."""
        agent = RoutingAgent(router_config)
        target = agent.route_workload('POOR_CHANNEL')
        assert target == 'DenosingAgent'

    def test_route_workload_good_channel(self, router_config):
        """Test workload routing for good channel."""
        agent = RoutingAgent(router_config)
        target = agent.route_workload('GOOD_CHANNEL')
        assert target == 'ExtrapolationAgent'

    def test_route_workload_fallback(self, router_config):
        """Test workload routing for fallback scenario."""
        agent = RoutingAgent(router_config)
        target = agent.route_workload('FALLBACK')
        assert target == 'LMMSEFallback'

    def test_route_workload_unknown_state(self, router_config):
        """Test workload routing with unknown channel state."""
        agent = RoutingAgent(router_config)
        target = agent.route_workload('UNKNOWN_STATE')
        assert target == 'LMMSEFallback'

    def test_snr_threshold_custom(self):
        """Test custom SNR threshold configuration."""
        config = {'snr_threshold_db': 10.0}
        agent = RoutingAgent(config)
        assert agent.snr_threshold_db == 10.0
        result = agent.evaluate_channel_state(rssi=-70, snr=9.5, doppler=100.0)
        assert result == 'POOR_CHANNEL'

    def test_doppler_threshold_custom(self):
        """Test custom Doppler threshold configuration."""
        config = {'doppler_max_hz': 500.0}
        agent = RoutingAgent(config)
        result = agent.evaluate_channel_state(rssi=-70, snr=10.0, doppler=450.0)
        assert result == 'GOOD_CHANNEL'
