# MinAtar
MinAtar is a testbed for AI agents which implements miniaturized version of several Atari 2600 games. MinAtar is inspired by the Arcade Learning Environment (Bellemare et. al. 2013), but simplifies the games to make experimentation with the environments more accessible and efficient. Currently, MinAtar provides analogues to five Atari games which play out on a 10x10 grid. The environments provide a 10x10xn state representation, where each of the n channels correspond to game-specific objects, such as ball, paddle and brick in the game Breakout.

<p  align="center">
<img src="img/seaquest.gif" width="200" />
<img src="img/breakout.gif" width="200" />
</p>
<p  align="center">
<img src="img/asterix.gif" width="200" />
<img src="img/freeway.gif" width="200" />
<img src="img/space_invaders.gif" width="200" />
</p>

## Quick Start
MinAtar consists of a python3 package, to use MinAtar follow the installation instructions. The included `DQN` and `AC_lambda` examples are written using `PyTorch`.

To install MinAtar simply:

```bash
git clone https://github.com/kenjyoung/MinAtar.git
# Configure a virtual environment (optional)
python3 -m venv venv
source venv/bin/activate
# Upgrade Pip
pip install --upgrade pip
# Install requirements
pip install -r requirements.txt
# Install
python setup.py install
# Ensure installation success
python examples/random_play.py -g breakout
# It should run for a bit and then you should see an output similar to:
# Avg Return: 0.5+/-0.023194827009486406
```

See examples/random_play.py for a simple example of how to use the module. To run this script do:

```bash
python random_play.py -g <game>
```

where `<game>` is one of the available games: asterix, breakout, freeway, seaquest and space_invaders. See the Games section below for details of each game. random_play.py will run 1000 episodes with a random policy and report the mean and standard error in the resulting returns.

To play a game as a human, run examples/human_play.py as follows:

```bash
python human_play.py -g <game>
```
Use the arrow keys to move and space bar to fire. Also press q to quit and r to reset.

Also included in the examples directory are example implementations of DQN and online actor-critic with eligibility traces.

## Results
The following plots display results for DQN (Mnih et al., 2015) and actor-critic with eligibility traces. Our DQN agent uses a significantly smaller network. We perform an ablation study of DQN, and display results for variants without experience replay, and without a seperate target network. Our AC agent uses a similar architecture to DQN, but does not use experience replay. We display results for two values of the trace decay parameter, 0.8 and 0.0.  Each curve is the average of 30 independent runs with different random seeds. For further information, see the paper on MinAtar available [here](https://arxiv.org/abs/1903.03176).

<img align="center" src="img/results.gif" width=800>

## Games
So far we have implemented analogues to five Atari games in MinAtar as follows. For each game we include a link to a video of a trained DQN agent playing.

### Asterix
The player can move freely along the 4 cardinal directions. Enemies and treasure spawn from the sides. A reward of +1 is given for picking up treasure. Termination occurs if the player makes contact with an enemy. Enemy and treasure direction are indicated by a trail channel. Difficulty is periodically increased by increasing the speed and spawn rate of enemies and treasure.

[Video](https://www.youtube.com/watch?v=Eg1XsLlxwRk)

### Breakout
The player controls a paddle on the bottom of the screen and must bounce a ball to break 3 rows of bricks along the top of the screen. A reward of +1 is given for each brick broken by the ball.  When all bricks are cleared another 3 rows are added. The ball travels only along diagonals. When the ball hits the paddle it is bounced either to the left or right depending on the side of the paddle hit. When the ball hits a wall or brick, it is reflected. Termination occurs when the ball hits the bottom of the screen. The ball's direction is indicated by a trail channel.

[Video](https://www.youtube.com/watch?v=cFk4efZNNVI&t)

### Freeway
The player begins at the bottom of the screen and the motion is restricted to traveling up and down. Player speed is also restricted such that the player can only move every 3 frames. A reward of +1 is given when the player reaches the top of the screen, at which point the player is returned to the bottom. Cars travel horizontally on the screen and teleport to the other side when the edge is reached. When hit by a car, the player is returned to the bottom of the screen. Car direction and speed is indicated by 5 trail channels.  The location of the trail gives direction while the specific channel indicates how frequently the car moves (from once every frame to once every 5 frames). Each time the player successfully reaches the top of the screen, the car speeds are randomized. Termination occurs after 2500 frames have elapsed.

[Video](https://www.youtube.com/watch?v=gbj4jiTcryw)

### Seaquest
The player controls a submarine consisting of two cells, front and back, to allow direction to be determined. The player can also fire bullets from the front of the submarine. Enemies consist of submarines and fish, distinguished by the fact that submarines shoot bullets and fish do not. A reward of +1 is given each time an enemy is struck by one of the player's bullets, at which point the enemy is also removed. There are also divers which the player can move onto to pick up, doing so increments a bar indicated by another channel along the bottom of the screen. The player also has a limited supply of oxygen indicated by another bar in another channel. Oxygen degrades over time, and is replenished whenever the player moves to the top of the screen as long as the player has at least one rescued diver on board. The player can carry a maximum of 6 divers. When surfacing with less than 6, one diver is removed. When surfacing with 6, all divers are removed and a reward is given for each active cell in the oxygen bar. Each time the player surfaces the difficulty is increased by increasing the spawn rate and movement speed of enemies. Termination occurs when the player is hit by an enemy fish, sub or bullet; or when oxygen reaches 0; or when the player attempts to surface with no rescued divers. Enemy and diver directions are indicated by a trail channel active in their previous location to reduce partial observability.

[Video](https://www.youtube.com/watch?v=W9k38b5QPxA&t)

### Space Invaders
The player controls a cannon at the bottom of the screen and can shoot bullets upward at a cluster of aliens above. The aliens move across the screen until one of them hits the edge, at which point they all move down and switch directions. The current alien direction is indicated by 2 channels (one for left and one for right) one of which is active at the location of each alien. A reward of +1 is given each time an alien is shot, and that alien is also removed. The aliens will also shoot bullets back at the player. When few aliens are left, alien speed will begin to increase. When only one alien is left, it will move at one cell per frame. When a wave of aliens is fully cleared, a new one will spawn which moves at a slightly faster speed than the last. Termination occurs when an alien or bullet hits the player.

[Video](https://www.youtube.com/watch?v=W-9Ru-RDEoI)

## Citing MinAtar
If you use MinAtar in your research please cite the following:

Young, K. Tian, T. (2019). MinAtar: An Atari-inspired Testbed for More Efficient Reinforcement Learning Experiments.  *arXiv preprint arXiv:1903.03176*.

In BibTeX format:

```
@Article{young19minatar,
author = {{Young}, Kenny and {Tian}, Tian},
title = {MinAtar: An Atari-inspired Testbed for More Efficient Reinforcement Learning Experiments},
journal = {arXiv preprint arXiv:1903.03176},
year = "2019"
}
```


## References
Bellemare, M. G., Naddaf, Y., Veness, J., & Bowling, M. (2013). The arcade learning environment: An evaluation platform for general agents. *Journal of Artificial Intelligence Research*, 47, 253–279.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., . . . others (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529.

## License
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
