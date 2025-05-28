import torch
import numpy as np
from gymnasium import spaces
# from typing import Literal # Not used after removing baseline agents
from .task_base import BaseTask
# from ..core.simulatior import AircraftSimulator # Not directly used by this task for enemy
from ..core.catalog import Catalog as c
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout # SafeReturn might be kept or removed based on new objectives
# Assuming you might want a condition for successful mission completion:
# from ..termination_conditions import MissionSuccessTermination # Placeholder for new termination
from ..reward_functions import AltitudeReward, EventDrivenReward # PostureReward removed
# from ..utils.utils import get_AO_TA_R, get2d_AO_TA_R, in_range_rad, LLA2NEU, get_root_dir # Utils for enemy interaction removed
from ..utils.utils import LLA2NEU # Keep if used for airbase coord transformation or other agent calcs
# BaselineActor and related agent logic removed
import logging # <<< ADD THIS IMPORT for logging messages

class AirbaseAttackTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.config = config # Ensure config is stored if BaseTask doesn't do it.

        # Define reward functions relevant to attacking an airbase
        self.reward_functions = [
            AltitudeReward(self.config),
            # EventDrivenReward might need adaptation or a new reward for airbase hits
            EventDrivenReward(self.config)
            # Add new reward functions specific to airbase attack if needed:
            # e.g., ProximityToAirbaseReward, AirbaseDamageReward
        ]

        # Define termination conditions
        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            Timeout(self.config),
            # Suggestion: Add an AirbaseDestroyedTermination condition here for explicit success.
            # e.g., AirbaseDestroyedTermination(self.config)
            # e.g., AgentTooFarFromObjectiveTermination(self.config)
        ]
        self._agent_die_flag = {}
        self._airbase_destroyed_reward_given = False # To ensure reward is given only once

        # Airbase properties (can also be loaded from config if they vary)
        self.airbase_attack_range = getattr(self.config, 'airbase_attack_range_m', 5000) # meters
        self.airbase_damage_per_hit = getattr(self.config, 'airbase_damage_per_hit', 10) # HP

    @property
    def num_agents(self) -> int:
        return 1

    def load_variables(self):
        # Define state variables for the agent
        self.state_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
            c.velocities_v_north_mps,
            c.velocities_v_east_mps,
            c.velocities_v_down_mps,
            c.velocities_u_mps,
            c.velocities_v_mps,
            c.velocities_w_mps,
            c.velocities_vc_mps,
            c.accelerations_n_pilot_x_norm,
            c.accelerations_n_pilot_y_norm,
            c.accelerations_n_pilot_z_norm,
            # Consider adding fuel status if relevant for mission duration
            # c.propulsion_fuel_available_lbs 
        ]
        # Define action variables
        self.action_var = [
            c.fcs_aileron_cmd_norm,
            c.fcs_elevator_cmd_norm,
            c.fcs_rudder_cmd_norm,
            c.fcs_throttle_cmd_norm,
            # Suggestion: Add weapon launch command to action_var for consistency
            # c.fcs_weapon_launch_cmd 
        ]
        # Define variables for rendering
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        # 9 ego states + 3 airbase relative states (dx, dy, dz to airbase) + 1 airbase status
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(13,), dtype=np.float32)


    def load_action_space(self):
        # Added discrete action for firing a weapon (0 for no fire, 1 for fire)
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30, 2]) # <<< MODIFIED HERE


    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.
        """
        sim = env.agents[agent_id]
        ego_props = sim.get_property_values(self.state_var)

        # (1) Ego agent info normalization
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[0] = ego_props[self.state_var.index(c.position_h_sl_m)] / 5000  # Altitude / 5km
        obs[1] = np.sin(ego_props[self.state_var.index(c.attitude_roll_rad)])
        obs[2] = np.cos(ego_props[self.state_var.index(c.attitude_roll_rad)])
        obs[3] = np.sin(ego_props[self.state_var.index(c.attitude_pitch_rad)])
        obs[4] = np.cos(ego_props[self.state_var.index(c.attitude_pitch_rad)])
        obs[5] = ego_props[self.state_var.index(c.velocities_u_mps)] / 340    # v_body_x / Mach
        obs[6] = ego_props[self.state_var.index(c.velocities_v_mps)] / 340    # v_body_y / Mach
        obs[7] = ego_props[self.state_var.index(c.velocities_w_mps)] / 340    # v_body_z / Mach
        obs[8] = ego_props[self.state_var.index(c.velocities_vc_mps)] / 340   # vc / Mach

        # (2) Airbase relative info
        airbase_data = env.airbase # This is the simple dict { "position": [lon, lat, alt], "hp": H, "destroyed": B }
        
        # Agent's current LLA position
        agent_lon, agent_lat, agent_alt = \
            ego_props[self.state_var.index(c.position_long_gc_deg)], \
            ego_props[self.state_var.index(c.position_lat_geod_deg)], \
            ego_props[self.state_var.index(c.position_h_sl_m)]

        # Airbase's LLA position (assuming airbase_data["position"] is [lon, lat, alt])
        airbase_lon, airbase_lat, airbase_alt = airbase_data["position"]
        
        # Convert to common NEU frame relative to a reference point
        agent_pos_neu = np.array(LLA2NEU(agent_lon, agent_lat, agent_alt, airbase_lon, airbase_lat, airbase_alt))
        airbase_pos_neu = np.array([0,0,0]) # Airbase is the origin in this relative NEU frame

        relative_neu = airbase_pos_neu - agent_pos_neu # Vector from agent to airbase
        
        obs[9] = relative_neu[0] / 10000  # Relative North / 10km
        obs[10] = relative_neu[1] / 10000 # Relative East / 10km
        obs[11] = relative_neu[2] / 10000 # Relative Down / 10km

        obs[12] = 1.0 if airbase_data["destroyed"] else 0.0

        return np.clip(obs, self.observation_space.low, self.observation_space.high)


    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value for JSBSim."""
        # Assumes action is a list/array: [aileron_idx, elevator_idx, rudder_idx, throttle_idx, fire_weapon_idx]
        norm_act = np.zeros(len(self.action_var)) 
        norm_act[0] = action[0] / 20.0  - 1.0  # aileron
        norm_act[1] = action[1] / 20.0 - 1.0   # elevator
        norm_act[2] = action[2] / 20.0 - 1.0   # rudder
        norm_act[3] = action[3] / (self.action_space.nvec[3]-1) * 0.5 + 0.4 # throttle (0.4 to 0.9)
        
        # Implement weapon fire action
        # Check if the action space has the fire command (i.e., len(action) is 5)
        if len(action) > 4: 
            weapon_fire_command = action[4] # 0 for no fire, 1 for fire
            
            # Check if agent has missiles left and if the airbase simulator exists in the environment
            if weapon_fire_command == 1 and env.agents[agent_id].num_left_missiles > 0 and hasattr(env, 'airbase_simulator'):
                logging.info(f"Agent {agent_id} fires a missile at airbase! Missiles left: {env.agents[agent_id].num_left_missiles}") # <<< MODIFIED HERE
                # Call the fire_missile method on the AircraftSimulator, passing the airbase_simulator
                missile_instance = env.agents[agent_id].fire_missile(target_sim=env.airbase_simulator) 
                if missile_instance:
                    env.add_temp_simulator(missile_instance)
            elif weapon_fire_command == 1 and env.agents[agent_id].num_left_missiles <= 0:
                logging.debug(f"Agent {agent_id} attempted to fire, but no missiles left.") # <<< MODIFIED HERE
            elif weapon_fire_command == 1 and not hasattr(env, 'airbase_simulator'):
                logging.warning("Agent attempted to fire missile, but airbase_simulator not found in environment.") # <<< MODIFIED HERE

        return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset and airbase state."""
        self._agent_die_flag = {agent_id: False for agent_id in env.agents.keys()}
        self._airbase_destroyed_reward_given = False
        
        # Reset airbase state (from env, which should handle its own reset)
        # Assuming env.reset() or env.reset_simulators() handles airbase dict reset
        if hasattr(env, 'airbase'):
            env.airbase["hp"] = getattr(self.config, 'airbase_initial_hp', 100)
            env.airbase["destroyed"] = False
        
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)
        # super().reset(env) # BaseTask.reset already calls reward_function.reset

    def step(self, env):
        """ Task-specific step logic, e.g. for airbase interaction.
            Called after simulator steps.
        """
        agent_id = list(env.agents.keys())[0] # Assuming single agent
        sim = env.agents[agent_id]
        airbase_data = env.airbase
        done_by_task = False # This flag is currently not used to terminate the episode here

        if not airbase_data["destroyed"]:
            # Simple "gun" or continuous damage if within range (remove if only missile based)
            agent_lla = sim.get_geodetic()
            airbase_lla = airbase_data["position"]
            
            # Calculate distance (approximate for LLA, or convert to NEU for better accuracy)
            R = 6371e3  # Earth radius in meters
            lat1, lon1 = np.radians(agent_lla[1]), np.radians(agent_lla[0])
            lat2, lon2 = np.radians(airbase_lla[1]), np.radians(airbase_lla[0])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
            c_dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            ground_dist = R * c_dist
            alt_diff = abs(agent_lla[2] - airbase_lla[2])
            dist_3d = np.sqrt(ground_dist**2 + alt_diff**2)

            if dist_3d < self.airbase_attack_range: 
                # This simulates direct attack like guns when in range
                # Uncomment and adjust if you want continuous gun damage:
                # airbase_data["hp"] -= self.airbase_damage_per_hit * env.dt 
                # logging.debug(f"Agent in attack range ({dist_3d:.0f}m). Airbase HP: {airbase_data['hp']:.1f}")
                pass 

            # Loop through launched missiles and log their positions
            for missile in sim.launch_missiles: # Corrected iteration for launched_missiles dictionary
                if missile.is_alive: # Log position only if the missile is still active/flying
                    missile_pos = missile.get_geodetic()
                    missile_velo = missile.get_velocity()
                    missile_coordinates = missile.get_position()

                    # Log missile position: Longitude, Latitude, Altitude (meters)
                    logging.debug(f"Missile {missile.uid} Position: Long={missile_pos[0]:.4f} deg, "
                                  f"Lat={missile_pos[1]:.4f} deg, Alt={missile_pos[2]:.2f} m")
                    logging.debug(f"Missile {missile.uid} Velocity: {missile_velo}")
                    logging.debug(f"Missile {missile.uid} Relative Position: {missile_coordinates}")

            # Check for missile hits from launched missiles
            for missile in sim.launch_missiles:
                # If missile has hit its target (status is HIT) and hasn't been processed for damage yet
                if missile.is_success and not getattr(missile, '_damage_applied', False): # Use is_success or !is_alive
                    if not airbase_data["destroyed"]:
                        print(f"Missile {missile.uid} hit airbase! HP before: {airbase_data['hp']}") # <<< MODIFIED HERE
                        airbase_data["hp"] -= getattr(self.config, 'missile_damage', 50) # Configurable missile damage
                        setattr(missile, '_damage_applied', True) # Mark damage as applied to prevent repeat damage
                        print(f"Airbase HP after: {airbase_data['hp']}") # <<< MODIFIED HERE
                
            if airbase_data["hp"] <= 0:
                airbase_data["hp"] = 0
                airbase_data["destroyed"] = True
                print("Airbase Destroyed by agent attack!") 
                # done_by_task = True # Uncomment this line if you want the task to force termination on destruction

        # Return True if task logic determines episode should end
        return done_by_task


    def get_reward(self, env, agent_id, info={}):
        """
        Calculate rewards.
        """
        reward = 0.0
        sim = env.agents[agent_id]
        airbase_data = env.airbase

        # Standard rewards from list (Altitude, EventDrivenReward)
        for reward_fn in self.reward_functions:
            reward += reward_fn.get_reward(self, env, agent_id)
        
        # Custom task rewards for airbase
        if not airbase_data["destroyed"]:
            pass # Damage reward is handled by EventDrivenReward if configured, or could be added explicitly here
        else: # Airbase is destroyed
            if not self._airbase_destroyed_reward_given:
                destruction_reward = getattr(self.config, 'airbase_destruction_reward', 200.0) #
                print(f"Airbase destroyed, granting reward: {destruction_reward}") 
                reward += destruction_reward
                self._airbase_destroyed_reward_given = True
        
        # Penalty for agent being dead
        if not sim.is_alive and not self._agent_die_flag[agent_id]:
            # This might be already handled by EventDrivenReward's crash/shotdown penalties
            # death_penalty = getattr(self.config, 'agent_death_penalty', -100.0)
            # reward += death_penalty
            # logging.info(f"Agent {agent_id} died, penalty: {death_penalty}")
            self._agent_die_flag[agent_id] = True
        
        info['airbase_hp'] = airbase_data["hp"]
        info['airbase_destroyed'] = airbase_data["destroyed"]
        return reward, info

    # get_termination is usually handled by the BaseTask by iterating self.termination_conditions.
    # If you add an AirbaseDestroyedTermination condition, it will be checked there.
    # Make sure such a condition is added to self.termination_conditions in __init__.
    # Example for a new termination condition (to be placed in termination_conditions folder):
    # class AirbaseDestroyedTermination(BaseTerminationCondition):
    #     def __init__(self, config):
    #         super().__init__(config)
    #     def get_termination(self, task, env, agent_id, info={}):
    #         if env.airbase["destroyed"]:
    #             info['success'] = True # Mark as successful termination
    #             return True, True, info
    #         return False, False, info