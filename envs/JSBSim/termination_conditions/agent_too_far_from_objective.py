from ..core.catalog import Catalog as c
from .termination_condition_base import BaseTerminationCondition
import logging

class AgentTooFar(BaseTerminationCondition):
    """
    End the simulation if the aircraft is too far from the waypoint.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_distance = getattr(config, 'max_waypoint_distance', 20000.0)  # meters (20 km default)

    def get_termination(self, task, env, agent_id, info={}):
        """
        Terminate if the agent is too far from the waypoint.
        """
        done = False
        success = False

        done = bool(env.agents[agent_id].get_property_value(c.detect_agent_too_far_waypoint_state))
        if done:
            env.agents[agent_id].crash()
            self.log(f'{agent_id} is too far from the waypoint! >{self.max_distance:.1f}m Total Steps={env.current_step}')

        return done, success, info    