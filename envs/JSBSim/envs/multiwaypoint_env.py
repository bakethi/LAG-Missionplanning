import math
from envs.JSBSim.human_task.HumanFreeFlyTask import HumanFreeFlyTask
from .env_base import BaseEnv
from ..tasks.multi_waypoint_task import MultiWaypointTask
from ..core.catalog import Catalog as c

class MultiWaypointEnv(BaseEnv):
    """
    MultiWaypointEnv is a fly-control env for single agent with no enemy fighters.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        assert len(self.agents) == 1, f"{self.__class__.__name__} only supports 1 aircraft"
        self.init_states = None
        self.active_waypoint = None
        self._task_stage = 0
        self.waypoint_sequence = {0: 100000001, 1: 100000002}  # fixed UID mapping

    @property
    def task_stage(self):
        return self._task_stage

    @task_stage.setter
    def task_stage(self, stage):
        if stage not in self.waypoint_sequence:
            raise ValueError(f"task_stage must be one of {list(self.waypoint_sequence.keys())}")
        self._task_stage = stage
        self.set_active_waypoint(self.waypoint_sequence[stage])

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'multiwaypoint':
            self.task = MultiWaypointTask(self.config)
        else:
            raise NotImplementedError(f'Unknown taskname: {taskname}')

    def reset(self):
        self.current_step = 0
        self.reset_simulators()
        self.heading_turn_counts = 0
        self.task.reset(self)
        self.task_stage = 0  # will automatically call set_active_waypoint
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]

        init_heading = self.np_random.uniform(0., 180.)
        init_altitude = self.np_random.uniform(14000., 30000.)
        init_velocities_u = self.np_random.uniform(400., 1200.)

        for init_state in self.init_states:
            init_state.update({
                'ic_psi_true_deg': init_heading,
                'ic_h_sl_ft': init_altitude,
                'ic_u_fps': init_velocities_u,
                'target_heading_deg': init_heading,
                'target_altitude_ft': init_altitude,
                'target_velocities_u_mps': init_velocities_u * 0.3048,
            })

        for idx, sim in enumerate(self.agents.values()):
            sim.reload(self.init_states[idx])

        self._tempsims.clear()

    def set_active_waypoint(self, wp_uid):
        self.active_waypoint = wp_uid
        for uid, wp in self.waypoints.items():
            wp.is_active = (uid == wp_uid)

        wp_sim = self.waypoints[wp_uid]
        wp_geod = wp_sim.get_geodetic()
        wp_alt = wp_geod[2]

        for sim in self.agents.values():
            sim.set_property_value(c.waypoint_longitude_geod_deg, wp_geod[0])
            sim.set_property_value(c.waypoint_latitude_geod_deg, wp_geod[1])
            sim.set_property_value(c.waypoint_altitude_ft, wp_alt / 0.3048)
            sim.set_property_value(c.detect_agent_reached_waypoint_state, False)
            sim.set_property_value(c.detect_agent_too_far_waypoint_state, False)
            

    def compute_distance_to_waypoint(self, agent_id):
        if not self.active_waypoint or self.active_waypoint not in self.waypoints:
            return float("inf")

        sim = self.agents[agent_id]
        wp_sim = self.waypoints[self.active_waypoint]

        agent_pos = sim.get_geodetic()
        wp_geod = wp_sim.get_geodetic()
        wp_lat = wp_geod[1]
        wp_lon = wp_geod[0]

        # Haversine formula
        phi1, phi2 = math.radians(agent_pos[1]), math.radians(wp_lat)
        dphi = math.radians(wp_lat - agent_pos[1])
        dlambda = math.radians(wp_lon - agent_pos[0])
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return 6371000 * c  # meters

    def get_alignment_to_waypoint(self, agent_id):
        if not self.active_waypoint or self.active_waypoint not in self.waypoints:
            return 0.0

        sim = self.agents[agent_id]
        wp_sim = self.waypoints[self.active_waypoint]
        agent_heading = sim.get_property_value(c.attitude_psi_rad)

        agent_pos = sim.get_position()[:2]
        wp_pos = wp_sim.get_position()[:2]

        delta = wp_pos - agent_pos
        wp_yaw = math.atan2(delta[1], delta[0])
        yaw_diff = (wp_yaw - agent_heading + math.pi) % (2 * math.pi) - math.pi
        return yaw_diff
