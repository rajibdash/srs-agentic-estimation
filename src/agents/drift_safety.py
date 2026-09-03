"""
Drift & Safety Agent: System Guardian & Reliability Overseer

Role: Monitors Block Error Rate (BLER) and CSI feedback loops. Validates
inference outputs against known performance baselines. If model performance
drops due to environment drift, marks samples for retraining and flags fallback
to traditional LMMSE.

Execution Target: DU Control Plane (ARM/CPU x86)
Telemetry Interval: 10 milliseconds
"""

from typing import Dict, Tuple
from collections import deque
import time


class DriftSafetyAgent:
    """Monitors model drift and ensures safe fallback mechanisms."""
    
    def __init__(self, config: dict):
        """
        Initialize the Drift & Safety Agent with safety thresholds.
        
        Args:
            config: Dictionary with max_evm_threshold, consecutive_failures_allowed, etc.
        """
        self.max_evm_threshold = config.get('max_evm_threshold', 0.18)
        self.consecutive_failures_allowed = config.get('consecutive_failures_allowed', 3)
        self.fallback_target = config.get('fallback_target', 'LMMSE_Hardware_Block')
        self.telemetry_interval_ms = config.get('telemetry_interval_ms', 10)
        
        # Monitoring state
        self.consecutive_failures = 0
        self.bler_history = deque(maxlen=1000)  # Track last 1000 BLER samples
        self.evm_history = deque(maxlen=1000)   # Track last 1000 EVM samples
        self.last_check_timestamp = time.time()
    
    def evaluate_model_performance(self, bler: float, evm: float) -> Tuple[bool, str]:
        """
        Evaluate model performance and determine if drift has occurred.
        
        Args:
            bler: Block Error Rate measurement
            evm: Error Vector Magnitude measurement
            
        Returns:
            Tuple of (is_healthy: bool, status_message: str)
        """
        self.bler_history.append(bler)
        self.evm_history.append(evm)
        
        if evm > self.max_evm_threshold:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.consecutive_failures_allowed:
                return False, f"DRIFT DETECTED: EVM {evm:.3f} exceeds threshold {self.max_evm_threshold}"
        else:
            self.consecutive_failures = 0  # Reset counter on success
        
        return True, "Model performance within acceptable bounds"
    
    def trigger_retraining(self) -> Dict:
        """
        Mark current batch for MLOps retraining pipeline.
        
        Returns:
            Retraining request metadata
        """
        return {
            'status': 'RETRAINING_TRIGGERED',
            'bler_samples': list(self.bler_history),
            'evm_samples': list(self.evm_history),
            'timestamp': time.time()
        }
    
    def trigger_fallback(self) -> str:
        """
        Trigger fallback to safe classical algorithm (LMMSE).
        
        Returns:
            Fallback target name
        """
        return self.fallback_target
    
    def get_telemetry(self) -> Dict:
        """Return current telemetry snapshot."""
        bler_avg = sum(self.bler_history) / len(self.bler_history) if self.bler_history else 0.0
        evm_avg = sum(self.evm_history) / len(self.evm_history) if self.evm_history else 0.0
        
        return {
            'timestamp': time.time(),
            'consecutive_failures': self.consecutive_failures,
            'bler_average': bler_avg,
            'evm_average': evm_avg,
            'bler_history_length': len(self.bler_history),
            'evm_history_length': len(self.evm_history)
        }
