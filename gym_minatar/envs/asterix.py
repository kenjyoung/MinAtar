#Adapted from https://github.com/qlan3/gym-games
import os
import importlib
import numpy as np
import gym
from gym import spaces

from gym_minatar.envs.base import BaseEnv

class AsterixEnv(BaseEnv):
  def __init__(self, display_time=50, **kwargs):
    self.game_name = 'asterix'
    self.display_time = display_time
    self.init(**kwargs)


if __name__ == '__main__':
  env = AsterixEnv()
  env.seed(0)
  print('Action space:', env.action_space)
  print('Action set:', env.action_set)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  for i in range(1):
    ob = env.reset()
    while True:
      action = env.action_space.sample()
      ob, reward, done, _ = env.step(action)
      print('Observation:', ob)
      print('Reward:', reward)
      print('Done:', done)
      if done:
        break
  env.close()