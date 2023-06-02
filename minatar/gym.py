# Adapted from https://github.com/qlan3/gym-games
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

try:
    import seaborn as sns
except:
    import logging
    logging.warning("Cannot import seaborn."
        "Will not be able to train from pixel observations.")

from minatar import Environment


class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "array", "rgb_array"]}

    def __init__(self, game, render_mode=None, display_time=50,
                use_minimal_action_set=False, **kwargs):
        self.render_mode = render_mode
        self.display_time = display_time

        self.game = Environment(env_name=game, **kwargs)

        if use_minimal_action_set:
            self.action_set = self.game.minimal_action_set()
        else:
            self.action_set = list(range(self.game.num_actions()))

        self.action_space = spaces.Discrete(len(self.action_set))
        self.observation_space = spaces.Box(
            0, 1, shape=self.game.state_shape(), dtype=np.uint8
        )

    def step(self, action):
        action = self.action_set[action]
        reward, done = self.game.act(action)
        if self.render_mode == "human":
            self.render()
        return self.game.state(), reward, done, False, {}

    def seed(self, seed=None):
        self.game.seed(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.game.reset()
        if self.render_mode == "human":
            self.render()
        return self.game.state(), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.render_mode == "array":
            return self.game.state()
        elif self.render_mode == "human":
            self.game.display_state(self.display_time)
        elif self.render_mode == "rgb_array": # use the same color palette of Environment.display_state
            state = self.game.state()
            n_channels = state.shape[-1]
            cmap = sns.color_palette("cubehelix", n_channels)
            cmap.insert(0, (0,0,0))
            numerical_state = np.amax(
                state * np.reshape(np.arange(n_channels) + 1, (1,1,-1)), 2)
            rgb_array = np.stack(cmap)[numerical_state]
            return rgb_array

    def close(self):
        if self.game.visualized:
            self.game.close_display()
        return 0


def register_envs():
    for game in ["asterix", "breakout", "freeway", "seaquest", "space_invaders"]:
        name = game.title().replace('_', '')
        register(
            id="MinAtar/{}-v0".format(name),
            entry_point="minatar.gym:BaseEnv",
            kwargs=dict(game=game, use_minimal_action_set=False),
        )
        register(
            id="MinAtar/{}-v1".format(name),
            entry_point="minatar.gym:BaseEnv",
            kwargs=dict(game=game, use_minimal_action_set=True),
        )
