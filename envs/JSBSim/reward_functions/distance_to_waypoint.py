import numpy as np
from .reward_function_base import BaseRewardFunction
import logging
from ..core.catalog import Catalog as c

class DistanceToWaypointReward(BaseRewardFunction):
    """
    Reward based on distance and heading alignment to the current active waypoint.
    Encourages the agent to get closer and face the target.
    """
    def __init__(self, config):
        super().__init__(config)
        self.success_radius = getattr(config, "success_radius", 500.0)  # meters
        self.max_distance = 70710.0  # battlefield diagonal (for normalization)
        self.punish_factor = 1.0     # scaling for distance penalty

    def get_reward(self, task, env, agent_id):
        distance = env.compute_distance_to_waypoint(agent_id)
        alignment_score = self.get_alignment_score_to_waypoint(env, agent_id)

        # Reward shaping
        if distance <= self.success_radius:
            distance_reward = 0.0
            logging.debug(f'agent[{agent_id}] reached waypoint within {distance:.1f}m')
        elif distance > 100000.0:
            distance_reward = -2.0  # extreme penalty
        else:
            # Closer = less negative
            distance_reward = -self.punish_factor * ((distance - self.success_radius) / self.max_distance)

        reward = distance_reward + alignment_score

        return self._process(reward, agent_id, (reward, distance, alignment_score))

    def get_alignment_score_to_waypoint(self, env, agent_id):
        yaw_diff = env.get_alignment_to_waypoint(agent_id)
        # Score ranges from 0 (aligned) to -1 (opposite)
        alignment_score = - (1 - np.cos(yaw_diff)) / 2
        return alignment_score





## comparing of reward shaping, hard and easy
# challenges, how i did it
# systematically, methodology