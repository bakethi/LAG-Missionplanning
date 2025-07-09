from ..core.catalog import Catalog as c
from .termination_condition_base import BaseTerminationCondition

class UnreachWaypoint(BaseTerminationCondition):
    """
    Terminates the episode if:
    - The agent fails to reach the current waypoint in time.
    - The agent strays too far from the current waypoint.
    - The agent reaches the final waypoint (success).
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_steps = getattr(self.config, 'max_steps', 2500)
        self.timeout_sec = self.max_steps * 0.2

    def get_termination(self, task, env, agent_id, info={}):
        done = False
        success = False
        cur_step = info.get('current_step', 0)
        sim_time = env.agents[agent_id].get_property_value(c.simulation_sim_time_sec)

        if env.task_stage is None:
            env.task_stage = 0

        # Check reach and too far for currently active waypoint
        reached_wp = env.agents[agent_id].get_property_value(c.detect_agent_reached_waypoint_state)
        too_far_wp = env.agents[agent_id].get_property_value(c.detect_agent_too_far_waypoint_state)

        # attitude checks
        # roll = env.agents[agent_id].get_property_value(c.attitude_roll_rad)
        # pitch = env.agents[agent_id].get_property_value(c.attitude_pitch_rad)
        # yaw_diff = abs(env.get_alignment_to_waypoint(agent_id))

        # level_threshold = 0.087 # 5 degrees

        # is_level = abs(roll) < level_threshold and abs(pitch) < level_threshold and yaw_diff < level_threshold

        # Case: reached current waypoint
        if reached_wp:
            if env.task_stage == len(env.waypoint_sequence) - 1:
                self.log(f'agent[{agent_id}] reached final waypoint {cur_step}, time {sim_time:.1f}s')
                print(f'agent[{agent_id}] reached final waypoint {env.task_stage} at step {cur_step}, time {sim_time:.1f}s')
                return True, True, info
            else:
                env.task_stage += 1
                print(f'agent[{agent_id}] reached waypoint {env.task_stage - 1} at step {cur_step}, time {sim_time:.1f}s')
                
                self.log(f'agent[{agent_id}] reached waypoint {env.task_stage - 1} at step {cur_step}, time {sim_time:.1f}s')
                env.set_active_waypoint(env.waypoint_sequence[env.task_stage])
                info['task_stage'] = env.task_stage
                return False, False, info

        # Case: timeout or too far
        if env.eval_flag:
            return False, False, info
        else:
            if sim_time > self.timeout_sec:
                self.log(f'agent[{agent_id}] timeout at step {cur_step}, time {sim_time:.1f}s')
                return True, False, info

        if too_far_wp:
            self.log(f'agent[{agent_id}] too far from waypoint at step {cur_step}, time {sim_time:.1f}s')
            return True, False, info

        return done, success, info
