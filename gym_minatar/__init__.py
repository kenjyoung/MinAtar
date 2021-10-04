from gym.envs.registration import register

for game in ['Asterix', 'Breakout', 'Freeway', 'Seaquest', 'SpaceInvaders']:
  register(
    id='{}-MinAtar-v0'.format(game),
    entry_point=f'gym_minatar.envs:{game}Env'
)