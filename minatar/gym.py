# Adapted from https://github.com/qlan3/gym-games
import gym
from gym import spaces
from gym.envs import register

from minatar import Environment

import numpy as np

channel2rgb = {
    0 : [255, 0, 0],
    1 : [0, 255, 0],
    2 : [0, 0, 255],
    3 : [128, 128, 0],
    4 : [128, 0, 128],
    5 : [0, 128, 128],
    6 : [170, 170, 85],
    7 : [170, 85, 170],
    8 : [85, 170, 170],
    9 : [85, 85, 170]
}

class BaseEnv(gym.Env):
    metadata = {"render.modes": ["human", "array"]}

    def __init__(self, game, display_time=50, use_minimal_action_set=False, **kwargs):
        self.game_name = game
        self.display_time = display_time

        self.game_kwargs = kwargs
        self.seed()

        if use_minimal_action_set:
            self.action_set = self.game.minimal_action_set()
        else:
            self.action_set = list(range(self.game.num_actions()))

        self.action_space = spaces.Discrete(len(self.action_set))
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=self.game.state_shape(), dtype=bool
        )

    def step(self, action):
        action = self.action_set[action]
        reward, done = self.game.act(action)
        return self.game.state(), reward, done, {}

    def reset(self):
        self.game.reset()
        return self.game.state()

    def seed(self, seed=None):
        self.game = Environment(
            env_name=self.game_name,
            random_seed=seed,
            **self.game_kwargs
        )
        return seed

    def render(self, mode="human"):
        if mode == "array":
            return self.game.state()
        elif mode == "human":
            self.game.display_state(self.display_time)
        elif mode == 'rgb_array':
            n_channels = self.game.n_channels
            state = self.game.state() # np.zeros((10, 10, n_channels), dtype=bool)
            array = np.zeros([100, 100, 3], dtype=np.uint8)
            for x in range(state.shape[0]):
                for y in range(state.shape[1]):
                    for l in range(n_channels):
                        if state[x, y, l] == True:
                            array[x*10:x*10+10, y*10:y*10+10] = channel2rgb[l]
            return array

    def close(self):
        if self.game.visualized:
            self.game.close_display()
        return 0


def register_envs():
    for game in ["asterix", "breakout", "freeway", "seaquest", "space_invaders"]:
        name = game.title().replace('_', '')
        register(
            id="{}-v0".format(name),
            entry_point="minatar.gym:BaseEnv",
            kwargs=dict(game=game, display_time=50, use_minimal_action_set=False),
        )
        register(
            id="{}-v1".format(name),
            entry_point="minatar.gym:BaseEnv",
            kwargs=dict(game=game, display_time=50, use_minimal_action_set=True),
        )
