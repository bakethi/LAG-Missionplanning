import numpy as np
from .reward_function_base import BaseRewardFunction
import logging
from ..core.catalog import Catalog as c 

class DistanceToBaseReward(BaseRewardFunction):
    """
    DistanceToWaypointReward
    Reward based on the distance and angle to the waypoint.
    - Zero reward for being at the waypoint.
    - Negative reward for being further away from the waypoint.
    """
    def __init__(self, config):
        super().__init__(config)
        # Standardize success radius across all components.
        # It was 500 in reward, 750 in ExtraCatalog. Let's pick 750 as it's larger.
        self.success_radius = getattr(self.config, "success_radius", 750.0) 
        self.max_distance = 70710.0 # Stays the same as before, assuming battlefield diagonal.
        
        # Increased punish_factor to make distance shaping more impactful
        self.punish_factor = getattr(self.config, "distance_punish_factor", 5.0) 
        # Added a factor for alignment score to make it more significant
        self.alignment_factor = getattr(self.config, "alignment_reward_factor", 10.0)

    def get_reward(self, task, env, agent_id):
        # Retrieve constants from self to ensure consistency and allow config override
        success_radius = self.success_radius
        max_distance = self.max_distance
        punish_factor = self.punish_factor
        alignment_factor = self.alignment_factor

        reward = 0.0

        # --- Waypoint Logic ---
        # Get the visited_waypoint flag from the environment.
        # This flag MUST be reset to False at the start of each episode in the environment's reset() method.
        visited_waypoint = getattr(env, "visited_waypoint", False)

        if not visited_waypoint:
            distance_wp = env.compute_distance_to_waypoint(agent_id)
            if distance_wp <= success_radius:
                logging.info(f'agent[{agent_id}] visited waypoint at {distance_wp:.1f}m')
                # Increased waypoint bonus significantly to make it a clear sub-goal
                reward += 500  # Bonus for visiting waypoint
                env.visited_waypoint = True # Mark as visited in the environment
            else:
                # Encourage reaching waypoint if not yet visited
                yaw_diff = env.get_alignment_to_waypoint(agent_id)
                alignment_score = self.get_alignment_score(yaw_diff)
                
                # Distance shaping towards waypoint
                # Ensure distance_wp - success_radius is not negative for the numerator if we are within the radius.
                # The expression (distance - success_radius) / max_distance already handles this naturally as it becomes negative/zero
                # when distance < success_radius, but the 'if distance_wp <= success_radius' block handles the positive reward.
                # So for the else block, distance_wp > success_radius.
                reward += -punish_factor * ((distance_wp - success_radius) / max_distance)
                
                # Apply scaled alignment score
                reward += alignment_score * alignment_factor
        else:
            # If waypoint is visited, we can optionally provide small positive reward for staying relatively close to the ideal path towards base,
            # or simply avoid penalizing for distance to waypoint once visited.
            # For simplicity, we just won't apply waypoint specific shaping once visited, as the base shaping will take over.
            pass


        # --- Base Logic ---
        distance_base = env.compute_distance_to_base(agent_id)
        
        if distance_base <= success_radius:
            # Base reward is higher if waypoint was visited
            base_reward = 1000.0 if visited_waypoint else 200.0 # Reduced direct-to-base reward for clearer distinction
            logging.info(f'agent[{agent_id}] reached base at {distance_base:.1f}m, reward = {base_reward}')
            reward += base_reward
        elif distance_base > 100000.0:
            # Punish severely for going too far from base (already implemented, but good to keep)
            reward -= 200.0
        else:
            # Encourage reaching base if not yet reached
            yaw_diff = env.get_alignment_to_base(agent_id)
            alignment_score = self.get_alignment_score(yaw_diff)
            
            # Distance shaping towards base
            reward += -punish_factor * ((distance_base - success_radius) / max_distance)
            
            # Apply scaled alignment score
            reward += alignment_score * alignment_factor

        return self._process(reward, agent_id, (reward, distance_base))

        
    def get_alignment_score(self, yaw_diff):
        # Range is 0 (aligned) to -1 (opposite), scaled by alignment_factor in get_reward
        return - (1 - np.cos(yaw_diff)) / 2