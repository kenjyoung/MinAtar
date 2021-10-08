#Adapted from https://github.com/qlan3/gym-games
from gym_minatar.envs.base import BaseEnv

class AsterixEnv(BaseEnv):
    def __init__(self, display_time=50, **kwargs):
        self.game_name = 'asterix'
        self.display_time = display_time
        self.init(**kwargs)

class MinimalActionAsterixEnv(BaseEnv):
    def __init__(self, display_time=50, **kwargs):
        self.game_name = 'asterix'
        self.display_time = display_time
        self.init(**kwargs, use_minimal_action_set=True)

class BreakoutEnv(BaseEnv):
    def __init__(self, display_time=50, **kwargs):
        self.game_name = 'breakout'
        self.display_time = display_time
        self.init(**kwargs)

class MinimalActionBreakoutEnv(BaseEnv):
    def __init__(self, display_time=50, **kwargs):
        self.game_name = 'breakout'
        self.display_time = display_time
        self.init(**kwargs, use_minimal_action_set=True)

class FreewayEnv(BaseEnv):
    def __init__(self, display_time=50, **kwargs):
        self.game_name = 'freeway'
        self.display_time = display_time
        self.init(**kwargs)

class MinimalActionFreewayEnv(BaseEnv):
    def __init__(self, display_time=50, **kwargs):
        self.game_name = 'freeway'
        self.display_time = display_time
        self.init(**kwargs, use_minimal_action_set=True)

class SeaquestEnv(BaseEnv):
    def __init__(self, display_time=50, **kwargs):
        self.game_name = 'seaquest'
        self.display_time = display_time
        self.init(**kwargs)

class MinimalActionSeaquestEnv(BaseEnv):
    def __init__(self, display_time=50, **kwargs):
        self.game_name = 'seaquest'
        self.display_time = display_time
        self.init(**kwargs, use_minimal_action_set=True)

class SpaceInvadersEnv(BaseEnv):
    def __init__(self, display_time=50, **kwargs):
        self.game_name = 'space_invaders'
        self.display_time = display_time
        self.init(**kwargs)

class MinimalActionSpaceInvadersEnv(BaseEnv):
    def __init__(self, display_time=50, **kwargs):
        self.game_name = 'space_invaders'
        self.display_time = display_time
        self.init(**kwargs, use_minimal_action_set=True)
