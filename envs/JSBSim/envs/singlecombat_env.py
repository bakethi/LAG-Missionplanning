import numpy as np
from .env_base import BaseEnv
from ..core.simulatior import StaticSimulator
from ..tasks import SingleCombatTask, SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, \
    HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask, AirbaseAttackTask


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 1, f"{self.__class__.__name__} now supports single agent scenarios!" # Modified line
        self.init_states = None
        if getattr(self.config, 'task', None) == 'op_story_01':
            # Initialize the airbase dictionary for HP and destroyed status
            # Use values from the config file (OpStory01.yaml)
            self.airbase = {
                "position": self.config.airbase_position, # Get position from config
                "hp": self.config.airbase_initial_hp,     # Get initial HP from config
                "destroyed": False
            }
                        # Create a StaticSimulator instance for the airbase
            airbase_uid = "Airbase01" # A unique ID for the airbase simulator
            airbase_model = "Airbase" # Model name for the airbase in rendering/logging
            airbase_color = "Blue"    # Team color, if applicable
            
            # Prepare init_state for StaticSimulator (expects meters for altitude if config uses meters)
            # Assuming airbase_position[2] from config is already in meters.
            airbase_init_state_m = {
                'ic_long_gc_deg': self.config.airbase_position[0],
                'ic_lat_geod_deg': self.config.airbase_position[1],
                'ic_h_sl_m': self.config.airbase_position[2] 
            }
            
            self.airbase_simulator = StaticSimulator(
                uid=airbase_uid,
                color=airbase_color,
                model=airbase_model,
                type="Ground+Static+Building", # Define its type for logging/rendering
                init_state=airbase_init_state_m,
                origin=getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)) # Use environment's origin
            )
            
            # Add the airbase_simulator to the environment's internal list of simulators (`self._jsbsims`).
            # This ensures it is managed (e.g., reset, rendered) by the BaseEnv's core loop.
            # `self._jsbsims` is already populated by `super().__init__()` (via `self.load_simulator()`).
            # We need to explicitly add our custom-created static simulator.
            if airbase_uid not in self._jsbsims:
                self._jsbsims[airbase_uid] = self.airbase_simulator
            else:
                # This case should ideally not happen if UIDs are unique or handled correctly.
                # It means a static base with this UID was already loaded, possibly from static_base_configs
                print(f"StaticSimulator with UID {airbase_uid} already exists in _jsbsims. Skipping addition.")
        else:
            # For other tasks, airbase might not be relevant
            self.airbase = {} 
            self.airbase_simulator = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'hierarchical_singlecombat':
            self.task = HierarchicalSingleCombatTask(self.config)
        elif taskname == 'singlecombat_dodge_missile':
            self.task = SingleCombatDodgeMissileTask(self.config)
        elif taskname == 'singlecombat_shoot':
            self.task = SingleCombatShootMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_dodge_missile':
            self.task = HierarchicalSingleCombatDodgeMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_shoot':
            self.task = HierarchicalSingleCombatShootTask(self.config)
        elif taskname == 'op_story_01':
            self.task = AirbaseAttackTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
            # switch side
            if self.init_states is None:
                self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
            # self.init_states[0].update({
            #     'ic_psi_true_deg': (self.np_random.uniform(270, 540))%360,
            #     'ic_h_sl_ft': self.np_random.uniform(17000, 23000),
            # })
            init_states = self.init_states.copy()
            #self.np_random.shuffle(init_states) # Removed shuffling
            for idx, sim in enumerate(self.agents.values()):
                sim.reload(init_states[idx % len(init_states)]) # Modified indexing
            self._tempsims.clear()
