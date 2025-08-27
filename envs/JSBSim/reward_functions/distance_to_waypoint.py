import numpy as np
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class DistanceToWaypointReward(BaseRewardFunction):
    """
    DistanceToWaypointReward:
    Encourages the agent to approach and align early with the active waypoint.
    - PBRS applied to waypoint proximity.
    - Gaussian reward for yaw alignment.
    """

    def __init__(self, config):
        super().__init__(config)
        self.success_radius = getattr(config, "success_radius", 750.0)
        self.max_distance = 141000.0
        self.punish_factor = getattr(config, "punish_factor", 1.0)
        self.yaw_sigma = getattr(config, "yaw_alignment_sigma", 0.25)  # width of Gaussian
        self.gamma = getattr(config, "gamma", 0.99)

        # Store last potential per agent (for PBRS)
        self.last_potential = {}

    def potential(self, distance):
        """
        Potential function φ(s).
        - Higher when closer to waypoint.
        - Normalized between 0 (far) and 1 (inside success radius).
        """
        if distance <= self.success_radius:
            return 1.0
        elif distance >= self.max_distance:
            return 0.0
        else:
            # Smoothly scale between 0 and 1
            return 1.0 - (distance - self.success_radius) / (self.max_distance - self.success_radius)

    def get_reward(self, task, env, agent_id):
        distance = env.compute_distance_to_waypoint(agent_id)
        yaw_diff = abs(env.get_alignment_to_waypoint(agent_id))  # radians

        # --- Proximity Reward with PBRS ---
        potential_now = self.potential(distance)
        potential_prev = self.last_potential.get(agent_id, potential_now)
        shaping_reward = self.gamma * potential_now - potential_prev
        self.last_potential[agent_id] = potential_now

        # Base proximity reward (goal reward)
        if distance <= self.success_radius:
            proximity_reward = 10.0
        else:
            proximity_reward = 0.0  # leave shaping to handle continuous incentive

        # --- Gaussian Alignment Reward ---
        alignment_score = np.exp(- (yaw_diff / self.yaw_sigma) ** 2)
        alignment_weight = 0.5 + 0.5 * (1 - min(distance / self.max_distance, 1.0))
        alignment_reward = alignment_score * alignment_weight

        # --- Total Reward ---
        total_reward = proximity_reward + shaping_reward + alignment_reward

        return self._process(total_reward, agent_id, (total_reward, distance, yaw_diff, alignment_score, shaping_reward))
