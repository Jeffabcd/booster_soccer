"""
Dummy runner for offline training without environment evaluation.
Returns empty logs to satisfy the training loop.
"""

from typing import Dict
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class DummyLowdimRunner(BaseLowdimRunner):
    """
    Dummy runner that does nothing.
    Use this when training offline without an environment for evaluation.
    """
    
    def __init__(self, *args, **kwargs):
        # Accept any arguments but don't use them
        pass
    
    def run(self, policy) -> Dict:
        """
        Returns empty logs without running any environment rollouts.
        """
        return {}
