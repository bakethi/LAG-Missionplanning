import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class ProximityToAirbaseReward(BaseRewardFunction):
    """
    Reward based on the proximity to the airbase.
    The closer the agent is to the airbase, the higher the reward.
    """
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_distance']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is inversely proportional to the distance from the airbase.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        # import enemy airbase position from catalog
        airbase = env.static_bases[100000001]
        airbase_lon = airbase.get_property_value(c.enemy_base_longitude_geod_deg)
        airbase_lat = airbase.get_property_value(c.enemy_base_latitude_geod_deg)

        agent_position = env.agents[agent_id].get_property_value(c.position_long_gc_deg), \
                         env.agents[agent_id].get_property_value(c.position_lat_geod_deg)

        distance = math.sqrt((airbase_lon - agent_position[0]) ** 2 +
                             (airbase_lat - agent_position[1]) ** 2)

        # Inverse distance reward
        if distance == 0:
            reward = float('inf')  # Maximum reward if at the airbase
        else:
            reward = 1 / distance  # Higher reward for closer proximity

        return self._process(reward, agent_id, (distance,))