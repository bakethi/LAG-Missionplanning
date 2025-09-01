import numpy as np
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class AltitudeReward(BaseRewardFunction):
    """
    Encourages agent to maintain its initial cruising altitude.
    Penalizes:
    - Deviations from that altitude.
    - Being too low (below danger/crash levels).
    - Unnecessary vertical motion.
    """

    def __init__(self, config):
        super().__init__(config)
        self.alt_tolerance_km = getattr(config, f'{self.__class__.__name__}_tolerance_km', 0.2)    # ±200m
        self.max_deviation_km = getattr(config, f'{self.__class__.__name__}_max_dev_km', 1.5)       # beyond this = full penalty
        self.altitude_limit = getattr(config, 'altitude_limit', 2500) / 1000  # km
        self.target_altitude_ft = 22000  # ft

        self.vertical_speed_penalty_weight = getattr(config, f'{self.__class__.__name__}_vz_weight', 0.3)
        self.max_penalty = 1.0

        self.reward_item_names = [self.__class__.__name__ + suffix for suffix in ['', '_Deviation', '_Vz', '_Crash']]

    def get_reward(self, task, env, agent_id):
        sim = env.agents[agent_id]

        # Altitude and vertical speed
        current_alt_km = sim.get_position()[-1] / 1000
        vertical_speed = sim.get_velocity()[-1] / 340  # Mach scale

        # Target altitude
        target_alt_km = self.target_altitude_ft * 0.0003048  # Convert ft to km

        # Reward for staying near target altitude
        deviation = abs(current_alt_km - target_alt_km)
        if deviation <= self.alt_tolerance_km:
            deviation_penalty = 0.0  # No penalty
        else:
            # Linear penalty up to max_deviation_km
            deviation_penalty = -self.max_penalty * min(1.0, (deviation - self.alt_tolerance_km) / self.max_deviation_km)

        # Penalty for vertical motion
        vz_penalty = -self.vertical_speed_penalty_weight * abs(vertical_speed)

        # Crash-level altitude
        crash_penalty = -50.0 if current_alt_km <= self.altitude_limit else 0.0

        reward = deviation_penalty + vz_penalty + crash_penalty
        return self._process(reward, agent_id, (deviation_penalty, vz_penalty, crash_penalty))
