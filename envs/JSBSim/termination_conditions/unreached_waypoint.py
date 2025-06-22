from ..core.catalog import Catalog as c
from .termination_condition_base import BaseTerminationCondition

class UnreachWaypoint(BaseTerminationCondition):
    """
    UnreachWaypoint [0, 1]
    End the simulation if the aircraft doesn't reach the waypoint within a limited time.
    """

    def __init__(self, config):
        super().__init__(config)
        self.success_radius = getattr(config, "success_radius", 500.0)  # meters
    
    def get_termination(self, task, env, agent_id, info={}):
        done = False
        success = False
        cur_step = info.get('current_step', 0)
        sim_time = env.agents[agent_id].get_property_value(c.simulation_sim_time_sec)
        
        success = env.agents[agent_id].get_property_value(c.detect_agent_reached_waypoint_state)

        if success:
            self.log(f'agent[{agent_id}] reached waypoint at step {cur_step}, time {sim_time:.1f}s')
            return True, True, info

        return done, success, info
