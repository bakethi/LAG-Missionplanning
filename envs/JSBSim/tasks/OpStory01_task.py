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
            # Add new termination conditions:
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
            # Add weapon launch command if not handled by a separate mechanism
            # c.fcs_weapon_launch_cmd # Placeholder for weapon launch
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
        # Adjust shape based on the actual features you decide on in get_obs
        # Example: 9 ego + 3 relative position to airbase + 1 airbase destroyed status = 13
        # If using delta_long, delta_lat, delta_alt, then it's 3 components.
        # If using relative NEU (North-East-Up) vector, it's also 3.
        # Plus agent's own altitude (1) + roll sin/cos (2) + pitch sin/cos (2) + v_body (3) + vc (1) = 9
        # Total 9 (ego) + 3 (airbase relative pos) + 1 (airbase destroyed status) = 13
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(13,), dtype=np.float32)


    def load_action_space(self):
        # Assuming same discrete action space as before for flight controls
        # aileron, elevator, rudder, throttle
        # Add a discrete action for firing a weapon if needed.
        # E.g., [41, 41, 41, 30, 2] where the last one is no-fire/fire
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])


    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.
        Observation:
        - Ego agent info (normalized)
            - [0] ego altitude         (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x         (unit: Mach speed, e.g., /340)
            - [6] ego v_body_y         (unit: Mach speed)
            - [7] ego v_body_z         (unit: Mach speed)
            - [8] ego_vc               (unit: Mach speed)
        - Airbase relative info
            - [9] relative_N_to_airbase (unit: 10km)
            - [10] relative_E_to_airbase (unit: 10km)
            - [11] relative_D_to_airbase (unit: 10km)  (Down, so positive if airbase is lower)
            - [12] airbase_destroyed_status (0 or 1)
        """
        sim = env.agents[agent_id]
        ego_props = sim.get_property_values(self.state_var) # Using self.state_var

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
        
        # Convert to common NEU frame relative to a reference point (e.g., agent's initial position or a fixed map origin)
        # For simplicity here, let's use the agent's current sim origin if available, or assume one
        # If env.lon0, env.lat0, env.alt0 are available from BaseEnv or SingleCombatEnv:
        # ref_lon, ref_lat, ref_alt = env.lon0, env.lat0, env.alt0 
        # Otherwise, use agent's initial position or a fixed point. Here, using airbase as ref for relative vector
        
        agent_pos_neu = np.array(LLA2NEU(agent_lon, agent_lat, agent_alt, airbase_lon, airbase_lat, airbase_alt))
        airbase_pos_neu = np.array([0,0,0]) # Airbase is the origin in this relative NEU frame

        relative_neu = airbase_pos_neu - agent_pos_neu # Vector from agent to airbase
        
        obs[9] = relative_neu[0] / 10000  # Relative North / 10km
        obs[10] = relative_neu[1] / 10000 # Relative East / 10km
        obs[11] = relative_neu[2] / 10000 # Relative Down (if airbase is at alt=0, this is -agent_alt_relative_to_airbase_alt) / 10km
                                          # Or more simply, (airbase_alt - agent_alt) / 10000

        obs[12] = 1.0 if airbase_data["destroyed"] else 0.0

        return np.clip(obs, self.observation_space.low, self.observation_space.high)


    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value for JSBSim."""
        # Assumes action is a list/array: [aileron_idx, elevator_idx, rudder_idx, throttle_idx]
        norm_act = np.zeros(len(self.action_var)) # Or fixed 4 if no weapon action
        norm_act[0] = action[0] / 20.0  - 1.0  # aileron
        norm_act[1] = action[1] / 20.0 - 1.0   # elevator
        norm_act[2] = action[2] / 20.0 - 1.0   # rudder
        norm_act[3] = action[3] / (self.action_space.nvec[3]-1) * 0.5 + 0.4 # throttle (0.4 to 0.9)
        # If you add a weapon fire action:
        # weapon_fire_command = action[4] # 0 for no fire, 1 for fire
        # if weapon_fire_command == 1 and env.agents[agent_id].num_left_missiles > 0:
        #     # Logic to designate airbase as target and launch missile
        #     # This is complex as MissileSimulator expects an AircraftSimulator target.
        #     # Simplification: task tells simulator to fire in direction of airbase.
        #     # Or, mark a flag that get_reward/step will use to check for "hit"
        #     print(f"Agent {agent_id} fires a missile!")
        #     env.agents[agent_id].fire_missile(target_coords=env.airbase["position"]) # Needs implementation in AircraftSimulator
        return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset and airbase state."""
        self._agent_die_flag = {agent_id: False for agent_id in env.agents.keys()}
        self._airbase_destroyed_reward_given = False
        
        # Reset airbase state (from env, which should handle its own reset)
        # Assuming env.reset() or env.reset_simulators() handles airbase dict reset
        # If not, do it here:
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
        done_by_task = False

        if not airbase_data["destroyed"]:
            # Simple "gun" or continuous damage if within range (remove if only missile based)
            agent_lla = sim.get_geodetic()
            airbase_lla = airbase_data["position"]
            
            # Calculate distance (approximate for LLA, or convert to NEU for better accuracy)
            # Simplified distance calculation (great-circle distance approximation for lat/lon)
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

            if dist_3d < self.airbase_attack_range: # e.g. 5km "gun" range
                # This simulates direct attack like guns when in range
                # For missile-only, remove this block and handle missile hits in get_reward
                # airbase_data["hp"] -= self.airbase_damage_per_hit * env.dt # Damage per second
                # print(f"Agent in attack range ({dist_3d:.0f}m). Airbase HP: {airbase_data['hp']:.1f}")
                pass # Placeholder for direct gun attack logic if any

            # Check for missile hits (simplistic: any successful missile hit damages airbase)
            # This assumes EventDrivenReward handles missile success by calling target.shotdown()
            # which we need to intercept or modify for a simple airbase.
            # A more robust way: iterate sim.launch_missiles, check their state and proximity to airbase.
            for missile in sim.launch_missiles:
                if missile.is_alive: # or a custom status for "hit simple target"
                    # Simple proximity check for missiles
                    missile_pos_neu = missile.get_position() # Assuming missile has NEU position from its origin
                    # Convert airbase LLA to NEU relative to missile's origin (sim.lon0, sim.lat0, sim.alt0)
                    airbase_neu_rel_missile_origin = LLA2NEU(*airbase_lla, sim.lon0, sim.lat0, sim.alt0)
                    
                    dist_missile_to_airbase = np.linalg.norm(missile_pos_neu - np.array(airbase_neu_rel_missile_origin))

                    if dist_missile_to_airbase < missile._Rc: # Using missile's explosion radius
                        if not airbase_data["destroyed"]:
                            print(f"Missile {missile.uid} hit airbase! HP before: {airbase_data['hp']}")
                            airbase_data["hp"] -= getattr(self.config, 'missile_damage', 50) # Configurable missile damage
                            missile.hit_target() # Ensure missile is marked as non-alive, exploded
                            print(f"Airbase HP after: {airbase_data['hp']}")
                
            if airbase_data["hp"] <= 0:
                airbase_data["hp"] = 0
                airbase_data["destroyed"] = True
                print("Airbase Destroyed by agent attack!")
                # done_by_task = True # This could be a success termination condition

        # Return True if task logic determines episode should end (e.g. objective met)
        # The main done signal will be aggregated by the environment from termination_conditions
        return done_by_task


    def get_reward(self, env, agent_id, info={}):
        """
        Calculate rewards:
        - Base rewards from reward_functions (Altitude, adapted EventDriven)
        - Task-specific rewards for airbase interaction.
        """
        reward = 0.0
        sim = env.agents[agent_id]
        airbase_data = env.airbase

        # Standard rewards from list (Altitude, EventDrivenReward)
        for reward_fn in self.reward_functions:
            reward += reward_fn.get_reward(self, env, agent_id)
        
        # Custom task rewards for airbase
        if not airbase_data["destroyed"]:
            # Reward for damaging airbase (if not covered by EventDrivenReward for missile hits)
            # This part might be redundant if EventDrivenReward correctly attributes missile hits
            # For example, if direct "gun" damage was applied in step():
            # if env.airbase["hp"] < self._previous_airbase_hp:
            #    reward += (self._previous_airbase_hp - env.airbase["hp"]) * some_multiplier
            # self._previous_airbase_hp = env.airbase["hp"] # Requires _previous_airbase_hp init
            pass
        else: # Airbase is destroyed
            if not self._airbase_destroyed_reward_given:
                destruction_reward = getattr(self.config, 'airbase_destruction_reward', 200.0)
                print(f"Airbase destroyed, granting reward: {destruction_reward}")
                reward += destruction_reward
                self._airbase_destroyed_reward_given = True
        
        # Penalty for agent being dead (crash or "shot down" by placeholder airbase defense)
        if not sim.is_alive and not self._agent_die_flag[agent_id]:
            # This might be already handled by EventDrivenReward's crash/shotdown penalties
            # death_penalty = getattr(self.config, 'agent_death_penalty', -100.0)
            # reward += death_penalty
            # print(f"Agent {agent_id} died, penalty: {death_penalty}")
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