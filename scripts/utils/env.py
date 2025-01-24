import gymnasium as gym
import copy
import random
import random
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Box, Lava
from minigrid.envs import ConfigWorldEnv, ConfigWorldEnvHavingKey
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, Wrapper
from configparser import ConfigParser
from .env_taxi import TaxiEnv

class TestWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(5)
        self.action_mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 5
        }
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, 64, 64),  # number of cells
            dtype="uint8",
        )
        new_spaces = self.observation_space.spaces.copy()
        new_spaces["image"] = new_image_space

    def step(self, action):
        if hasattr(action, 'item'):
            action = action.item()
        mapped_action = self.action_mapping[action]
        return self.env.step(mapped_action)

    def observation(self, obs):
        return {**obs, "image": obs['image']}

def read_maps_from_config(config_file='random_maps.config'):
    config = ConfigParser()
    config.read(config_file, encoding='UTF-8')
    
    maps = []
    if 'maps' not in config.sections():
        raise ValueError("No [maps]")
    
    # for key in sorted(config['random_maps'].keys()):
    for i in range(1, len(config['maps']) + 1):
        map_str = config['maps'][f"map{i}"]
        map_lines = map_str.strip().split('\n')
        map_grid = []
        for line in map_lines:
            row = [cell.strip() for cell in line.strip().split(',')]
            map_grid.append(row)
        maps.append(map_grid)
    
    return maps

class CustomMinigridEnv(MiniGridEnv):
    """
    ## Registered Configurations

    - `MiniGrid-RandomEnvWrapper-v0`
    """

    def __init__(self, size=8, random_num = 1, config_path='random_maps.config', max_steps: int | None = 256, curriculum = 1, fixed_map = None, havekey = False, **kwargs):
        self.env_key = "MiniGrid"
        self.size=size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
        )
        self.config_path = config_path
        self.maps = read_maps_from_config(config_path)
        self.size = len(self.maps[0])
        self.height = len(self.maps[0])
        self.width = len(self.maps[0])
        self.curriculum = curriculum
        self.fixed_map = fixed_map
        self.havekey = havekey
        self.random_num = random_num
        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            # **kwargs,
        )
    @staticmethod
    def _gen_mission():
        return (
            "get the key from the room, "
            "unlock the door and "
            "go to the goal"
        )

    def _gen_grid(self, width, height):
        if self.curriculum == 1:
            selected_map = random.choice(self.maps[:self.random_num])
        elif self.curriculum == 2:
            selected_map = random.choice(self.maps[self.random_num:2*self.random_num])
        else:
            selected_map = random.choice(self.maps[2*self.random_num:3*self.random_num])
        if self.fixed_map is not None:
            selected_map = self.maps[self.fixed_map]
        
        self.selected_map = selected_map
        self.grid = Grid(self.width, self.height)

        if self.havekey:
            self.grid.set(0,0, Key("blue"))
            self.carrying = Key("blue")

        for i in range(1,len(selected_map)):
            for j in range(0,len(selected_map)):
                if(selected_map[i][j]== 'x'):
                    self.grid.set(j, i, Wall())

                if(selected_map[i][j]== 'S'):
                    self.agent_pos = self.place_agent(
                        top=(j, i), size=(1, 1)
                    )

                if(selected_map[i][j]== 'G'):
                    self.grid.set(j,i, Goal())

                if(selected_map[i][j]== 'D'):
                    # colors = set(COLOR_NAMES)
                    # color = self._rand_elem(sorted(colors))
                    self.grid.set(j,i, Door("blue", is_locked=True))

                if(selected_map[i][j]== 'K' and not self.havekey):
                    self.grid.set(j,i, Key("blue"))

                if(selected_map[i][j]== 'E'):
                    self.grid.vert_wall(j,i,1, Lava)

                if(selected_map[i][j]== 'B'):
                    self.grid.set(j,i, Box("blue",Key("blue")))

        self.mission = self._gen_mission()
    
    def observation(self, obs):
        return {**obs, "image": obs['image']}
   

def make_env(env_key, seed=None,  max_steps=256, curriculum=1, fixed_map=None,render_mode="rgb_array", config_path="configmap.config"):
    if curriculum == 3:
        havekey = False
    else:
        havekey = True

    if env_key == "MiniGrid-ConfigWorld-Random":
        env = CustomMinigridEnv(config_path=config_path, random_num=5, max_steps=max_steps, curriculum=curriculum, fixed_map=fixed_map, havekey=havekey)
    elif env_key == "MiniGrid-ConfigWorld-v0":
        env = CustomMinigridEnv(config_path=config_path, random_num=1, max_steps=max_steps, curriculum=curriculum, fixed_map=fixed_map, havekey=havekey)
    elif env_key == "Taxi-v0":
        env = TaxiEnv(max_steps=max_steps)
        # env = gym.make(env_key, render_mode=render_mode)
    # env = TestWrapper(env)
    env.reset(seed=seed)
    return env

def copy_env(copied_env, env_key, seed=None, render_mode="rgb_array", curriculum=1, **kwargs):
    if env_key == "Taxi-v0":
        copied_env.reset()
        return copied_env
    env_code = copied_env.grid.encode()
    curriculum = copied_env.curriculum
    max_steps = copied_env.max_steps
    fixed_map = copied_env.fixed_map
    config_path = copied_env.config_path
    # havekey = copied_env.havekey
    # random_num = copied_env.random_num
    # new_env = gym.make(env_key, render_mode=render_mode)
    new_env = make_env(env_key, seed=seed, render_mode=render_mode, curriculum=curriculum, max_steps=max_steps, fixed_map=fixed_map, config_path=config_path)
    new_env.grid, _ = new_env.grid.decode(env_code)
    new_env.agent_pos = copy.deepcopy(copied_env.agent_pos)
    new_env.agent_dir = copy.deepcopy(copied_env.agent_dir)
    new_env.reset(seed=seed)
    return new_env