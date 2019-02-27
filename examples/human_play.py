################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
#                                                                                                              #
# python3 human_test.py -g <game>                                                                              #
################################################################################################################

import argparse
import tkinter as Tk
from minatar import Environment, GUI

################################################################################################################
# Script that allows a human to play any of the MinAtar games. Use arrow keys to move and space to fire. 
# Pressing q will exit the game, r will restart.
#
################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--game", "-g", type=str)
args = parser.parse_args()

# Setup game environment and GUI
env = Environment(args.game)
gui = GUI(env.game_name(), env.n_channels)

# Thread safe variables for use with GUI
action = Tk.IntVar()
action.set(0)
G = Tk.DoubleVar()
G.set(0.0)
is_terminate = Tk.BooleanVar()
is_terminate.set(False)

# Key handler for human player
def on_key_event(event):
    if event.key == "q":  # quit the game
        gui.quit()
    elif (event.key  == 'r'): # reset the game environment
        env.reset()
    elif (event.key  == ' '): # shoots bullet
        action.set(5)
    elif (event.key == 'left'):
        action.set(1)
    elif (event.key == 'up'):
        action.set(2)
    elif (event.key  == 'right'):
        action.set(3)
    elif (event.key  == 'down'):
        action.set(4)

################################################################################################################
# play
#
# Allows user to the play the game and displays state and score via gui.
#
################################################################################################################
def play():
    if is_terminate.get() == True:
        gui.quit()

    a = action.get()
    r, t = env.act(a)
    is_terminate.set(t)
    G.set(G.get()+r)
    gui.display_state(env.state())
    gui.set_message("Score: " + str(G.get()))
    action.set(0)
    gui.update(50, play)

# Hook up the key handler and initiate game play
gui.overwrite_key_handle(on_key_event)
gui.update(0, play)
gui.run()
print("Final Score: "+str(G.get()))
