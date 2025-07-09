import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class StabilityReward(BaseRewardFunction):
    """
    StabilityReward:
    Encourages stable flight by penalizing roll/pitch and angular rates.
    Yaw rate is only penalized when not actively turning.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_penalty = getattr(config, 'stability_max_penalty', 0.2)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_attitude', '_rate']]

    def get_reward(self, task, env, agent_id):
        sim = env.agents[agent_id]
        
        # Get current aircraft attitude (orientation in space)
        roll = sim.get_property_value(c.attitude_roll_rad)        # Bank angle: positive = right wing down
        pitch = sim.get_property_value(c.attitude_pitch_rad)      # Nose up/down: positive = nose up
        
        # Get current angular rates (how fast the aircraft is rotating)
        roll_rate = sim.get_property_value(c.velocities_p_rad_sec)   # Roll rate: how fast we're banking
        pitch_rate = sim.get_property_value(c.velocities_q_rad_sec)  # Pitch rate: how fast nose moves up/down
        yaw_rate = sim.get_property_value(c.velocities_r_rad_sec)    # Yaw rate: how fast we're turning left/right

        # Calculate how far off course we are (0 = perfectly aligned, π = opposite direction)
        heading_error = abs(env.get_alignment_to_waypoint(agent_id))  # Range: [0, π]
        
        # Create a turn factor: 0 = flying straight, 1 = need major course correction
        # If heading error > 60°, we're definitely in a turn maneuver
        turn_factor = min(heading_error / math.radians(60), 1.0)      # Range: [0, 1]

        # Dynamic tolerances: Be more lenient during turns, strict during straight flight
        roll_tol = 0.2 + 0.3 * turn_factor    # Straight: 0.2 rad (11°), Turn: 0.5 rad (29°)
        rate_tol = 0.1 + 0.3 * turn_factor    # Straight: 0.1 rad/s, Turn: 0.4 rad/s

        # Calculate attitude stability reward (closer to 1.0 = better stability)
        # Uses Gaussian decay: perfect = 1.0, gets worse exponentially as angles increase
        attitude_r = math.exp(-((roll / roll_tol) ** 2 + (pitch / 0.3) ** 2))
        
        # Calculate rate stability reward (closer to 1.0 = smoother flight)
        # Penalizes jerky movements and rapid attitude changes
        rate_r = math.exp(-((roll_rate / rate_tol) ** 2 + (pitch_rate / 0.3) ** 2))

        # Only penalize yaw rate when we're already aligned with waypoint
        # This prevents oscillating/hunting behavior when on correct heading
        if heading_error < math.radians(5):  # Within 5° of correct heading
            rate_r *= math.exp(-((yaw_rate / 0.3) ** 2))

        # Convert stability measures to penalty: perfect stability = 0 penalty, poor = max penalty
        # attitude_r * rate_r ranges from 0 (terrible) to 1 (perfect)
        # So (1 - attitude_r * rate_r) ranges from 0 (perfect) to 1 (terrible)
        stability_penalty = -self.max_penalty * (1 - attitude_r * rate_r)

        # Reduce stability requirements near waypoints to avoid conflicts with arrival rewards
        # Far away: full penalty, close (<1000m): reduced penalty
        distance = env.compute_distance_to_waypoint(agent_id)
        distance_scale = min(distance / 1000.0, 1.0)  # Scale from 0.0 to 1.0
        stability_penalty *= distance_scale

        return self._process(stability_penalty, agent_id, (attitude_r, rate_r))
