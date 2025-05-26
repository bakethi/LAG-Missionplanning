from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c

class AirBaseDestroyedTermination(BaseTerminationCondition):
    """
    AirBaseDestroyedTermination
    End the simulation if the airbase is destroyed.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the airbase is destroyed.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        done = bool(env.agents[agent_id].get_property_value(c.detect_enemy_base_state))
        if done:
            env.agents[agent_id].crash()
            self.log(f'{agent_id} has been destroyed! Total Steps={env.current_step}')
        success = False
        return done, success, info