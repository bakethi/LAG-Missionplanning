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
        self.reach_radius = getattr(config, "waypoint_reach_radius", 500.0)
        self.timeout_sec = getattr(config, "waypoint_timeout_sec", 500)

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

        # Case: reached current waypoint
        if reached_wp:
            print("reached a waypoint")
            if env.task_stage == 0:
                env.task_stage = 1
                info['task_stage'] = 1
                env.set_active_waypoint(env.waypoint_sequence[1])
                self.log(f'agent[{agent_id}] reached waypoint 0 at step {cur_step}, time {sim_time:.1f}s')
                return False, False, info

            elif env.task_stage == 1:
                self.log(f'agent[{agent_id}] reached final waypoint at step {cur_step}, time {sim_time:.1f}s')
                return True, True, info

        # Case: timeout or too far
        if sim_time > self.timeout_sec:
            self.log(f'agent[{agent_id}] timeout at step {cur_step}, time {sim_time:.1f}s')
            return True, False, info

        if too_far_wp:
            self.log(f'agent[{agent_id}] too far from waypoint at step {cur_step}, time {sim_time:.1f}s')
            return True, False, info

        return done, success, info
