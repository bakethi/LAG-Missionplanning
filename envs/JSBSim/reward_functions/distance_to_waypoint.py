import numpy as np
from .reward_function_base import BaseRewardFunction
import logging
from ..core.catalog import Catalog as c 

class DistanceToWaypointReward(BaseRewardFunction):
    """
    DistanceToWaypointReward
    Reward based on the distance and angle to the waypoint.
    - Zero reward for being at the waypoint.
    - Negative reward for being further away from the waypoint.
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        distance = env.compute_distance_to_waypoint(agent_id)

        # Define a "success" radius and battlefield max diagonal
        success_radius = getattr(self.config, "success_radius", 500.0)  # meters
        max_distance = 70710.0  # battlefield diagonal ~ sqrt(50km^2 + 50km^2)

        Pv = 0.
        punish_factor = 1.0

        if distance <= success_radius:
            Pv = +0.0
            logging.info(f'agent[{agent_id}] reached waypoint at distance {distance:.1f}m')
        elif distance > 100000.0:
            Pv = -2.0
        else:
            Pv = -punish_factor * ((distance - success_radius) / max_distance)

        # alignment
        alignment_score = self.get_alignment_score_to_waypoint(env, agent_id) # 0 for facing the waypoint, -1 for facing opposite        
        Pv += alignment_score
        #logging.info(f'agent[{agent_id}] distance to waypoint: {distance:.1f}m, Pv: {Pv:.3f}, alignment: {alignment_score:.3f}')

        reward = Pv
        return self._process(reward, agent_id, (Pv, distance))
    
    def get_alignment_score_to_waypoint(self, env, agent_id):
        yaw_diff = env.get_alignment_to_waypoint(agent_id)

        alignment_score = - (1 - np.cos(yaw_diff)) / 2  # 0 when aligned, -1 when opposite

        return alignment_score




## compairing of reaward shaping, hard and easy
# challenges, how i did it
# systematically, methodology