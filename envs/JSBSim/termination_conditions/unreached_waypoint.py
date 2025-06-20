from ..core.catalog import Catalog as c
from .termination_condition_base import BaseTerminationCondition

class UnreachWaypoint(BaseTerminationCondition):
    """
    UnreachWaypoint [0, 1]
    End the simulation if the aircraft doesn't reach the waypoint within a limited time.
    """

    def __init__(self, config):
        super().__init__(config)
        self.success_radius = getattr(config, "waypoint_success_radius", 500.0)  # meters
        self.max_time = getattr(config, "waypoint_max_time", 100.0)  # seconds

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        End the simulation if the aircraft didn't reach the waypoint in limited time.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        done = False
        success = False
        cur_step = info.get('current_step', 0)
        sim_time = env.agents[agent_id].get_property_value(c.simulation_sim_time_sec)

        if sim_time > self.max_time:
            done = True
            self.log(f'agent[{agent_id}] did not reach waypoint within {self.max_time:.1f}s at step {cur_step}, time {sim_time:.1f}s')
            env.agents[agent_id].crash()
            return done, success, info

        success = env.agents[agent_id].get_property_value(c.detect_agent_reached_waypoint_state)
        if success:
            self.log(f'agent[{agent_id}] reached waypoint at step {cur_step}, time {sim_time:.1f}s')
            # crash the agent if it reached the waypoint
            env.agents[agent_id].crash()
            return True, True, info

        return done, success, info