import numpy as np
from .env_base import BaseEnv
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
        self.airbase = {
            "position": [120.0, 60.0, 0.0],  # Example:  Replace with desired coordinates
            "hp": 100,
            "destroyed": False
        }

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
