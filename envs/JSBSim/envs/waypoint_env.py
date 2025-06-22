from envs.JSBSim.human_task.HumanFreeFlyTask import HumanFreeFlyTask
from .env_base import BaseEnv
from ..tasks.reach_waypoint_task import ReachWaypointTask
from ..core.catalog import Catalog as c 
import math


class WaypointEnv(BaseEnv):
    """
    WaypointEnv is a fly-control env for single agent with no enemy fighters.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 1, f"{self.__class__.__name__} only supports 1 aircraft"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'waypoint':
            self.task = ReachWaypointTask(self.config)
        elif taskname == "HumanFreeFly":
            self.task = HumanFreeFlyTask(self.config)
        else:
            raise NotImplementedError(f'Unknown taskname: {taskname}')

    def load_waypoints(self):
        waypoint_configs = getattr(self.config, 'waypoint_configs', None)
        self._waypoints = []
        if waypoint_configs is not None:
            for wp_id, wp_cfg in waypoint_configs.items():
                color = wp_cfg.get('color', 'Green')
                type = wp_cfg.get('type', 'Navaid+Static+Waypoint')
                init_state = wp_cfg.get('init_state', {})

                position = self.geodetic_to_neu(
                    init_state.get('waypoint/longitude-geod-deg', 120.1),
                    init_state.get('waypoint/latitude-geod-deg', 60.1),
                    init_state.get('waypoint/altitude-ft', 4000.0) * 0.3048
                )

                self._waypoints.append({
                    'uid': wp_id,
                    'color': color,
                    'type': type,
                    'position': position,
                    'init_state': init_state,
                    'geod_position': (init_state.get('waypoint/longitude-geod-deg', 120.1),
                                      init_state.get('waypoint/latitude-geod-deg', 60.1),
                                      init_state.get('waypoint/altitude-ft', 4000.0) * 0.3048)
                })

    @property
    def waypoints(self):
        """
        Get the list of waypoints.
        Returns:
            list: List of waypoints with their properties.
        """
        return self._waypoints

    def render_waypoint(self, uid, position, color, model, type_, filepath=None):
        """
        Render a waypoint to the ACMI file.
        Args:
            uid (str): Unique ID for the waypoint
            position (tuple): (north, east, up) in meters (NEU coordinates)
            color (str): Color string
            model (str): Model name
            type_ (str): Type string
        """
        # Convert NEU to geodetic if needed, or use directly
        # Example: lon, lat, alt = NEU2LLA(*position, self.lon0, self.lat0, self.alt0)
        lon, lat, alt = self.neu_to_geodetic(*position) 

        # Compose ACMI line (Tacview format example)
        line = (
            f"{uid},T={lon}|{lat}|{alt}|0|0|0,"
            f"Type=Type=Navaid+Static+Waypoint,Color={color}\n"
        )

        # Write to ACMI file (append mode)
        with open(filepath, "a") as f:
            f.write(line)

    def load_waypoints_into_simulator(self):
        """
        Load the first waypoint into the simulator property catalog.
        """
        self.load_waypoints()  # Load from config if not already

        if not self._waypoints:
            return

        # Set the first waypoint (only one supported for now)
        wp = self._waypoints[0]
        for sim in self.agents.values():  # Assuming all agents use same waypoint
            sim.set_property_value(c.waypoint_longitude_geod_deg, wp["geod_position"][0])
            sim.set_property_value(c.waypoint_latitude_geod_deg, wp["geod_position"][1])
            sim.set_property_value(c.waypoint_altitude_ft, wp["geod_position"][2] / 0.3048)

    def neu_to_geodetic(self, north, east, up):
        # Implement or import this conversion as needed
        from ..utils.utils import NEU2LLA
        return NEU2LLA(north, east, up, 120.0, 60.0, 0.0)

    def geodetic_to_neu(self, lon, lat, alt):
        # Implement or import this conversion as needed
        from ..utils.utils import LLA2NEU
        return LLA2NEU(lon, lat, alt, 120.0, 60.0, 0.0)
    
    def compute_distance_to_waypoint(self, agent_id):
        """
        Compute the distance from the agent to the nearest waypoint.
        Args:
            agent_id (str): The ID of the agent.
        Returns:
            float: Distance to the nearest waypoint in meters.
        """
        # distance in 2D (ignoring altitude)
        self.load_waypoints()
        agent_position = self.agents[agent_id].get_geodetic()
        if not self.waypoints:
            return float('inf')
        # haversine formula to calculate distance
        # Extract lat/lon in degrees
        agent_lat = agent_position[1]  # latitude in degrees
        agent_lon = agent_position[0]  # longitude in degrees
        wp_lat = self.waypoints[0]['geod_position'][1]  # waypoint latitude in degrees
        wp_lon =   self.waypoints[0]['geod_position'][0]  # waypoint longitude in degrees

        # Convert to radians
        phi1 = math.radians(agent_lat)
        phi2 = math.radians(wp_lat)
        dphi = math.radians(wp_lat - agent_lat)
        dlambda = math.radians(wp_lon - agent_lon)

        # Haversine formula
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        R = 6371000  # Earth radius in meters
        distance = R * c

        return distance
    
    def get_alignment_to_waypoint(self, agent_id):
        """
        Get the alignment of the agent to the nearest waypoint.
        Args:
            agent_id (str): The ID of the agent.
        Returns:
            float: Alignment score, 0 when facing the waypoint, -1 when facing opposite.
        """
        agent = self.agents[agent_id]
        agent_heading = agent.get_property_value(c.attitude_psi_rad)
        # Extract waypoint position
        waypoint_position = self.waypoints[0]['position'][:2]  # Only
        # x/y or north/east
        agent_position = agent.get_position()[:2]  # Only x/y or north/e
        delta = waypoint_position - agent_position
        waypoint_yaw = math.atan2(delta[1], delta[0])
        yaw_diff = waypoint_yaw - agent_heading
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        
        return yaw_diff 


    def reset(self):
        self.current_step = 0
        self.reset_simulators()
        self.heading_turn_counts = 0
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        init_heading = self.np_random.uniform(0., 180.)
        # -> initial longitude and latitude: in heading.yaml ic_lat_geod_deg 60; ic_long_gc_deg 120
        # -> current position can be obtained by sim.get_property_values([c.position_lat_geod_deg, c.position_long_gc_deg])
        # -> target position can be obtained by sim.get_property_values([c.target_latitude_geod_deg, c.target_longitude_geod_deg])
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
        self.load_waypoints_into_simulator()