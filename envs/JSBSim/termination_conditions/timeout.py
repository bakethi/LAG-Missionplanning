from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c


class Timeout(BaseTerminationCondition):
    """
    Timeout
    Episode terminates if max_step steps have passed.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_steps = getattr(self.config, 'max_steps', 2500)
        # Track time spent within waypoint radius for each agent
        self.waypoint_timeout_start = {}  # agent_id -> step when entered radius
        self.waypoint_timeout_threshold = 10.0  # seconds
        self.waypoint_radius = 1000  # meters
        self.within_waypoint_radius = {}  # agent_id -> bool (stays True until waypoint reached)

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        Terminate if max_step steps have passed or agent circles waypoint too long

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        if env.eval_flag:
            return False, False, info
        else:
            # Check if agent has properly reached waypoint (using catalog flag)
            reached_waypoint = env.agents[agent_id].get_property_value(c.detect_agent_reached_waypoint_state)
            # roll = env.agents[agent_id].get_property_value(c.attitude_roll_rad)
            # pitch = env.agents[agent_id].get_property_value(c.attitude_pitch_rad)
            # yaw_diff = abs(env.get_alignment_to_waypoint(agent_id))

            # level_threshold = 0.087 # 5 degrees

            # is_level = abs(roll) < level_threshold and abs(pitch) < level_threshold and yaw_diff < level_threshold

            if reached_waypoint:
                # Agent reached waypoint properly - reset tracking
                self.within_waypoint_radius[agent_id] = False
                if agent_id in self.waypoint_timeout_start:
                    del self.waypoint_timeout_start[agent_id]
            else:
                # Check distance to waypoint
                distance = env.compute_distance_to_waypoint(agent_id)
                current_time = env.current_step * 0.2
                
                if distance <= self.waypoint_radius:
                    # Agent is within waypoint radius
                    if not self.within_waypoint_radius.get(agent_id, False):
                        # First time entering radius - set flag and start timer
                        self.within_waypoint_radius[agent_id] = True
                        self.waypoint_timeout_start[agent_id] = current_time
                    else:
                        # Already been in radius - check timeout
                        time_in_radius = current_time - self.waypoint_timeout_start[agent_id]
                        if time_in_radius >= self.waypoint_timeout_threshold:
                            self.log(f"{agent_id} failed - spent {time_in_radius:.1f}s circling waypoint")
                            return True, False, info  # done=True, success=False
                # Note: We don't reset the flag when agent goes outside radius
                # It stays True until waypoint is properly reached
            
            # Check regular timeout
            done = env.current_step >= self.max_steps
            if done:
                self.log(f"{agent_id} step limits! Total Steps={env.current_step}")
            success = False
            return done, success, info
