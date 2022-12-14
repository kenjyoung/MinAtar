# Adapted from https://github.com/qlan3/gym-games
import gym
from gym import spaces
from gym.envs import register

from minatar import Environment


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
        return self.game.state(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        if(seed is not None):
            self.game = Environment(
                env_name=self.game_name,
                random_seed=seed,
                **self.game_kwargs
            )
        self.game.reset()
        return self.game.state(), {}

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
