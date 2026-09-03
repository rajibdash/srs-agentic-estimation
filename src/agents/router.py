"""
Routing Agent: Traffic Controller & Context Sensor

Role: Acts as an instantaneous gating function. Analyzes inbound raw SRS parameters
and groups them cleanly into environment profiles. If a profile falls below safety
or noise thresholds, it routes to the appropriate specialized agent.

Execution Target: FPGA / SmartNIC (inline)
"""

import time
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class RoutingAgent:
    """Coordinates workload routing based on channel conditions."""
    
    def __init__(self, config: dict):
        """
        Initialize the Routing Agent with SNR and Doppler thresholds.
        
        Args:
            config: Dictionary containing SNR threshold and Doppler max frequency
        """
        self.snr_threshold_db = config.get('snr_threshold_db', 12.5)
        self.doppler_max_hz = config.get('doppler_max_hz', 300)
        self.execution_target = config.get('execution_target', 'FPGA_Inline')
        
        # Latency tracking for O-RAN compliance
        self._routing_times = []
        self._max_timing_samples = 100
    
    def evaluate_channel_state(self, rssi: float, snr: float, doppler: float) -> str:
        """
        Evaluate incoming SRS channel conditions and determine routing decision.
        Includes latency instrumentation for O-RAN timing budget validation.
        
        Args:
            rssi: Received Signal Strength Indicator
            snr: Signal-to-Noise Ratio in dB
            doppler: Doppler spread in Hz
            
        Returns:
            Routing decision: 'POOR_CHANNEL' | 'GOOD_CHANNEL' | 'FALLBACK'
        """
        start_time = time.perf_counter()
        
        if snr < self.snr_threshold_db:
            result = 'POOR_CHANNEL'
        elif doppler > self.doppler_max_hz:
            result = 'POOR_CHANNEL'
        else:
            result = 'GOOD_CHANNEL'
        
        # Record latency
        elapsed_us = (time.perf_counter() - start_time) * 1e6
        self._routing_times.append(elapsed_us)
        if len(self._routing_times) > self._max_timing_samples:
            self._routing_times.pop(0)
        
        return result
    
    def route_workload(self, channel_state: str) -> str:
        """
        Route SRS signal to appropriate processing agent.
        
        Args:
            channel_state: Channel state classification
            
        Returns:
            Target agent name
        """
        routing_map = {
            'POOR_CHANNEL': 'DenosingAgent',
            'GOOD_CHANNEL': 'ExtrapolationAgent',
            'FALLBACK': 'LMMSEFallback'
        }
        return routing_map.get(channel_state, 'LMMSEFallback')
    
    def get_latency_metrics(self) -> Dict:
        """
        Get routing latency metrics for O-RAN compliance monitoring.
        
        Returns:
            Dictionary with latency statistics
        """
        if not self._routing_times:
            return {
                'avg_routing_latency_us': 0.0,
                'max_routing_latency_us': 0.0,
                'num_routing_calls': 0
            }
        
        import numpy as np
        return {
            'avg_routing_latency_us': float(np.mean(self._routing_times)),
            'max_routing_latency_us': float(np.max(self._routing_times)),
            'num_routing_calls': len(self._routing_times)
        }
