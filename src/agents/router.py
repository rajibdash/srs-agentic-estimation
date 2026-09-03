"""
Routing Agent: Traffic Controller & Context Sensor

Role: Acts as an instantaneous gating function. Analyzes inbound raw SRS parameters
and groups them cleanly into environment profiles. If a profile falls below safety
or noise thresholds, it routes to the appropriate specialized agent.

Execution Target: FPGA / SmartNIC (inline)
"""

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
    
    def evaluate_channel_state(self, rssi: float, snr: float, doppler: float) -> str:
        """
        Evaluate incoming SRS channel conditions and determine routing decision.
        
        Args:
            rssi: Received Signal Strength Indicator
            snr: Signal-to-Noise Ratio in dB
            doppler: Doppler spread in Hz
            
        Returns:
            Routing decision: 'POOR_CHANNEL' | 'GOOD_CHANNEL' | 'FALLBACK'
        """
        if snr < self.snr_threshold_db:
            return 'POOR_CHANNEL'
        
        if doppler > self.doppler_max_hz:
            return 'POOR_CHANNEL'
        
        return 'GOOD_CHANNEL'
    
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
