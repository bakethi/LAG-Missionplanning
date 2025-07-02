import numpy as np
from .reward_function_base import BaseRewardFunction

class AltitudeReward(BaseRewardFunction):
    """
    AltitudeReward:
    Penalizes the agent for:
    - Descending below a safe altitude with significant vertical speed.
    - Flying below a danger altitude.
    - Severely penalizes if below a critical crash altitude.
    """

    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(config, f'{self.__class__.__name__}_safe_altitude', 4.0)      # km
        self.danger_altitude = getattr(config, f'{self.__class__.__name__}_danger_altitude', 3.5)  # km
        self.altitude_limit = getattr(config, 'altitude_limit', 2500) / 1000                       # km
        self.Kv = getattr(config, f'{self.__class__.__name__}_Kv', 0.2)                            # descent scale
        self.max_penalty = 1.0

        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]

    def get_reward(self, task, env, agent_id):
        altitude_km = env.agents[agent_id].get_position()[-1] / 1000
        vertical_speed_mach = env.agents[agent_id].get_velocity()[-1] / 340  # < 0 if descending

        Pv = 0.0
        if altitude_km <= self.safe_altitude and vertical_speed_mach < 0:
            descent_ratio = (self.safe_altitude - altitude_km) / self.safe_altitude
            Pv = -self.max_penalty * min(1.0, -vertical_speed_mach / self.Kv * descent_ratio)

        PH = 0.0
        if altitude_km <= self.danger_altitude:
            PH = -self.max_penalty * (1.0 - altitude_km / self.danger_altitude)

        if altitude_km <= self.altitude_limit:
            PH = -30.0  # severe penalty on crash

        reward = Pv + PH
        return self._process(reward, agent_id, (Pv, PH))
