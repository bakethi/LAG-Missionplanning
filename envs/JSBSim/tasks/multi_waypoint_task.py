import numpy as np
from gymnasium import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, AttitudeReward, StabilityReward
from ..reward_functions import DistanceToWaypointReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout
from ..termination_conditions import AgentTooFar , UnreachWaypoint


class MultiWaypointTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            AltitudeReward(self.config),
            DistanceToWaypointReward(self.config),
            StabilityReward(self.config),
        ]
        self.termination_conditions = [
            #UnreachHeading(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config), 
            AgentTooFar(self.config),
            Timeout(self.config),
            UnreachWaypoint(self.config),
        ]

    @property
    def num_agents(self):
        return 1

    def load_variables(self):
        # Only include used state variables
        self.state_var = [
            c.position_h_sl_m,          # 0. altitude (unit: m)
            c.attitude_roll_rad,        # 1. roll (unit: rad)
            c.attitude_pitch_rad,       # 2. pitch (unit: rad)
            c.velocities_u_mps,         # 3. v_body_x (unit: m/s)
            c.velocities_v_mps,         # 4. v_body_y (unit: m/s)
            c.velocities_w_mps,         # 5. v_body_z (unit: m/s)
            c.velocities_vc_mps,        # 6. vc (unit: m/s)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,     # [-1., 1.]
            c.fcs_elevator_cmd_norm,    # [-1., 1.]
            c.fcs_rudder_cmd_norm,      # [-1., 1.]
            c.fcs_throttle_cmd_norm,    # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(11,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.

        Observation (dim 11):
            0. altitude (5km)
            1. roll_sin
            2. roll_cos
            3. pitch_sin
            4. pitch_cos
            5. v_body_x (mach)
            6. v_body_y (mach)
            7. v_body_z (mach)
            8. vc (mach)
            9. distance to current waypoint (normalized)
            10. alignment to current waypoint (rad)
        """
        obs = np.array(env.agents[agent_id].get_property_values(self.state_var))
        norm_obs = np.zeros(11)

        norm_obs[0] = obs[0] / 5000         # altitude (5km)
        norm_obs[1] = np.sin(obs[1])        # roll_sin
        norm_obs[2] = np.cos(obs[1])        # roll_cos
        norm_obs[3] = np.sin(obs[2])        # pitch_sin
        norm_obs[4] = np.cos(obs[2])        # pitch_cos
        norm_obs[5] = obs[3] / 340          # v_body_x (mach)
        norm_obs[6] = obs[4] / 340          # v_body_y (mach)
        norm_obs[7] = obs[5] / 340          # v_body_z (mach)
        norm_obs[8] = obs[6] / 340          # vc (mach)

        # Waypoint metrics
        distance = env.compute_distance_to_waypoint(agent_id)
        alignment = env.get_alignment_to_waypoint(agent_id)
        norm_obs[9] = distance / 141000     # battlefield diagonal normalization
        norm_obs[10] = alignment            # radians

        return np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
    
    
    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return norm_act