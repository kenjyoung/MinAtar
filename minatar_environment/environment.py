################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
################################################################################################################
from importlib import import_module
import numpy as np


#####################################################################################################################
# Environment
#
# Wrapper for all the specific game environments. Imports the environment specified by the user and then acts as a
# minimal interface. Also defines code for displaying the environment for a human user. 
#
#####################################################################################################################
class Environment:
    def __init__(self, env_name, sticky_action_prob = 0.1, difficulty_ramping = True, random_seed = None):
        env_module = import_module('minatar.environments.'+env_name)
        self.env_name = env_name
        self.env = env_module.Env(ramping = difficulty_ramping, seed = random_seed)
        self.n_channels = self.env.state_shape()[2]
        self.sticky_action_prob = sticky_action_prob
        self.last_action = 0

    # Wrapper for env.act
    def act(self, a):
        if(np.random.rand()<self.sticky_action_prob):
            a = self.last_action
        self.last_action = a
        return self.env.act(a)

    # Wrapper for env.state
    def state(self):
        return self.env.state()

    # Wrapper for env.reset
    def reset(self):
        return self.env.reset()

    # Wrapper for env.state_shape
    def state_shape(self):
        return self.env.state_shape()

    # All MinAtar environments have 6 actions
    def num_actions(self):
        return 6

    def game_name(self):
        return self.env_name

    def minimal_action_set(self):
        return self.env.minimal_action_set()
