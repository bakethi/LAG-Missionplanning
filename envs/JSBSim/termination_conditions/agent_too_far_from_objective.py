from ..core.catalog import Catalog as c
from .termination_condition_base import BaseTerminationCondition

class AgentTooFar(BaseTerminationCondition):
    """
    Terminates the episode if the agent flies too far from the current active waypoint.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_distance = getattr(config, 'max_waypoint_distance', 141000.0)  # meters

    def get_termination(self, task, env, agent_id, info={}):
        done = False
        success = False

        # Ensure task_stage is initialized
        task_stage = env.task_stage if env.task_stage is not None else 0

        # Always use the active waypoint's "too far" flag
        too_far = bool(env.agents[agent_id].get_property_value(c.detect_agent_too_far_waypoint_state))

        if too_far:
            self.log(f'agent[{agent_id}] is too far from waypoint {task_stage} (> {self.max_distance:.1f}m) | Step={env.current_step}')
            info['too_far'] = True
            done = True

        return done, success, info
