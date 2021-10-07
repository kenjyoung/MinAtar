################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
#                                                                                                              #
# python3 agent_play.py -g <game> -f <directory/file name prefix> -n <number of runs> -w <window size>         #
#   -a, --agent: which agent type? currently DQN or AC                                                         #                                                            #
################################################################################################################

import torch
import argparse, numpy

from minatar import Environment
from minatar.gui import GUI
from dqn import QNetwork, get_state
from AC_lambda import ACNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################################################################################################
# find_best_run
#
# Go through all the files of the runs and select network parameters based on the best final moving average.
#
# Input:
#   file_name: directory of data files plus the file name prefix (e.g., directory/seaquest)
#   num_runs: number of runs given by the user
#   window_size: specify the size of the moving window for averaging the average returns
#
# Output: best parameters of the network
#
#################################################################################################################
def find_best_run(file_name, num_runs, window_size, agent):

    returns_runs_episodes = []
    for i in numpy.arange(1, num_runs+1):
        data_and_weights = torch.load(file_name + "_" + str(i) + "_data_and_weights",
                                      map_location=lambda storage, loc: storage)

        returns = data_and_weights['returns']
        returns_runs_episodes.append(returns)

    # Get smoothed values and find the best last moving averages
    max_mv_avg = 0.0
    best_run_index = 0
    for index, val in enumerate(returns_runs_episodes):
        mv_avg_per_run = numpy.convolve(val, numpy.ones((window_size,)) / window_size, mode='valid')

        if mv_avg_per_run[-1] > max_mv_avg:
            max_mv_avg = mv_avg_per_run[-1]
            best_run_index = index

    best_weights = torch.load(file_name + "_" + str(best_run_index+1) + "_data_and_weights",
                              map_location=lambda storage, loc: storage)
    if(agent=='DQN'):
        return best_weights['policy_net_state_dict']
    elif(agent=='AC'):
        return best_weights['network_state_dict']
    else:
        raise ValueError('Unknown agent type')


################################################################################################################
# class agent
#
# A wrapper for a trained agent that controls game play and interfaces with the GUI
#
# Input:
#   env: the environment of a game
#   network_params: optimal parameters of the network
#   which_agent: DQN or AC
#   gui: reference a GUI
#
################################################################################################################
class agent:
    def __init__(self, env, network_params, which_agent, gui):
        self.in_channels = env.n_channels
        self.num_actions = env.num_actions()
        self.env = env
        self.which_agent = which_agent

        if (which_agent == 'DQN'):
            self.network = QNetwork(self.in_channels, self.num_actions).to(device)
        elif (which_agent == 'AC'):
            self.network = ACNetwork(self.in_channels, self.num_actions).to(device)
        self.network.load_state_dict(network_params)
        self.network.eval()

        # Display related
        self.gui = gui

        self.initialize()

    # Defines the policy of the agent w.r.t the trained network.
    def select_action(self):
        if (self.which_agent == 'DQN'):
            # Greedy behavior policy for action selection
            with torch.no_grad():
                action = self.network(self.s).max(1)[1].view(1, 1)
        elif (self.which_agent == 'AC'):
            # Draw action from multinomial specified by policy network
            with torch.no_grad():
                action = torch.multinomial(self.network(self.s)[0], 1)[0]
        else:
            raise ValueError('Unknown agent type')
        return action

    # Initialize the game
    def initialize(self):
        # Initialize the environment and start state
        self.env.reset()
        self.s = get_state(self.env.state())
        self.is_terminated = False

        self.G = 0

    # Play a game
    def play(self):
        self.gui.display_state(self.env.state())

        if self.is_terminated == True:
            self.gui.set_message("Game over! Score: " + str(self.G))
            self.initialize()
            self.gui.update(3000, self.play)
            return

        self.gui.set_message("Score: " + str(self.G))
        action = self.select_action()
        reward, self.is_terminated = self.env.act(action)
        self.G += reward
        s_prime = get_state(self.env.state())

        # Progress the game
        self.s = s_prime
        self.gui.update(50, self.play)


################################################################################################################
# run_agent
#
# Main program that sets up a GUI and a trained agent with the best parameters.  Then, it runs the agent with
# game visualization hosted inside the GUI.
#
# Input:
#   env: the environment of a game
#   network_params: optimal parameters of the network
#   which_agent: DQN or AC
#
################################################################################################################
def run_agent(env, network_params, which_agent):
    # Setup the display
    gui = GUI(env.game_name(), env.n_channels)

    # Setup agent
    sim_agent = agent(env, network_params, which_agent, gui)

    # Initiate the game play
    gui.update(0, sim_agent.play)
    gui.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", type=str)
    parser.add_argument("--agent", "-a", type=str, default="DQN")
    parser.add_argument("--filename", "-f", type=str)
    parser.add_argument("--windowsize", "-w", type=str)
    parser.add_argument("--numruns", "-n", type=str)
    args = parser.parse_args()

    env = Environment(args.game)

    network_param = find_best_run(args.filename, int(args.numruns), int(args.windowsize), args.agent)
    run_agent(env, network_param, args.agent)


if __name__ == '__main__':
    main()
