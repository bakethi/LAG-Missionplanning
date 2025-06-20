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
        agent_position = env.agents[agent_id].get_position()[:2]      # Only x/y or north/east
        waypoint_position = env.waypoints[0]['position'][:2]          # Only x/y or north/east
        distance = np.linalg.norm(agent_position - waypoint_position)  # meters

        # Define a "success" radius and battlefield max diagonal
        success_radius = 100.0
        max_distance = 70710.0  # battlefield diagonal ~ sqrt(50km^2 + 50km^2)

        Pv = 0.
        punish_factor = 1.0

        if distance <= success_radius:
            Pv = +0.0
            logging.info(f'agent[{agent_id}] reached waypoint at distance {distance:.1f}m')
        elif distance > 200000.0:
            Pv = -2.0
        else:
            Pv = -punish_factor * ((distance - success_radius) / max_distance)

        # alignment
        alignment_score = self.get_alignment_to_waypoint(env, agent_id) # 0 for facing the waypoint, -1 for facing opposite        
        Pv += alignment_score
        #logging.info(f'agent[{agent_id}] distance to waypoint: {distance:.1f}m, Pv: {Pv:.3f}, alignment: {alignment_score:.3f}')



        reward = Pv
        return self._process(reward, agent_id, (Pv, distance))


    def compute_distance_to_waypoint(self, env, agent_id):
        env.load_waypoints()  # Ensure waypoints are loaded
        agent_position = env.agents[agent_id].get_position()[:2]
        waypoint_position = env.waypoints[0]['position'][:2]
        distance = np.linalg.norm(agent_position - waypoint_position)
        return distance
    
    def compute_angle_to_waypoint(self, env, agent_id):
        agent = env.agents[agent_id]
        agent_position =  agent.get_position()[:2]
        waypoint_position = env.waypoints[0]['position'][:2]
        delta_x = waypoint_position[0] - agent_position[0]
        delta_y = waypoint_position[1] - agent_position[1]
        angle = np.arctan2(delta_y, delta_x)
        return angle
    
    def get_alignment_to_waypoint(self, env, agent_id):
        agent = env.agents[agent_id]
        agent_heading = agent.get_property_value(c.attitude_psi_rad)  # yaw in radians

        waypoint_position = env.waypoints[0]['position'][:2]
        agent_position = agent.get_position()[:2]

        delta = waypoint_position - agent_position
        waypoint_yaw = np.arctan2(delta[1], delta[0])

        yaw_diff = waypoint_yaw - agent_heading
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]

        alignment_score = - (1 - np.cos(yaw_diff)) / 2  # 0 when aligned, -1 when opposite

        return alignment_score




## compairing of reaward shaping, hard and easy
# challenges, how i did it
# systematically, methodology