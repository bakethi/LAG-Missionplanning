import numpy as np
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class DistanceToWaypointReward(BaseRewardFunction):
    """
    DistanceToWaypointReward:
    Encourages the agent to approach and align early with the active waypoint.
    - Penalizes straying far from the waypoint.
    - Rewards smaller yaw difference (alignment) more heavily near perfect alignment.
    """

    def __init__(self, config):
        super().__init__(config)
        self.success_radius = getattr(config, "success_radius", 750.0)
        self.max_distance = 141000.0
        self.punish_factor = getattr(config, "punish_factor", 1.0)
        self.yaw_sigma = getattr(config, "yaw_alignment_sigma", 0.25)  # width of Gaussian

    def get_reward(self, task, env, agent_id):
        distance = env.compute_distance_to_waypoint(agent_id)
        yaw_diff = abs(env.get_alignment_to_waypoint(agent_id))  # radians

        # --- Proximity Reward ---
        if distance <= self.success_radius:
            proximity_reward = 10.0
        elif distance > self.max_distance:
            proximity_reward = -1.0
        else:
            # Smooth squared shaping
            proximity_reward = -self.punish_factor * ((distance - self.success_radius) / self.max_distance) ** 2

        # --- Gaussian Alignment Reward ---
        alignment_score = np.exp(- (yaw_diff / self.yaw_sigma) ** 2)

        # Optional: scale alignment reward to emphasize it more when closer
        alignment_weight = 0.5 + 0.5 * (1 - min(distance / self.max_distance, 1.0))
        alignment_reward = alignment_score * alignment_weight

        # --- Total Reward ---
        total_reward = proximity_reward + alignment_reward

        return self._process(total_reward, agent_id, (total_reward, distance, yaw_diff, alignment_score))
