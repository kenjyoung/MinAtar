################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
#                                                                                                              #
# python3 human_test.py -g <game>                                                                              #
################################################################################################################

import argparse
import tkinter as Tk
from minatar import Environment
from minatar.gui import GUI

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
action_taken = Tk.BooleanVar()
action_taken.set(False)
action_released = Tk.BooleanVar()
action_released.set(False)
G = Tk.DoubleVar()
G.set(0.0)
is_terminate = Tk.BooleanVar()
is_terminate.set(False)

# Map input keys to agent actions
key_action_map = {' ':5, 'left':1, 'up':2, 'right':3, 'down':4}

# Key press handler for human player
def on_key_event(event):
    if event.key == 'q':  # quit the game
        gui.quit()
    elif (event.key  == 'r'): # reset the game environment
        env.reset()
    elif(event.key in key_action_map):
        key_action = key_action_map[event.key]
        action.set(key_action)
        # When new action is selected it has not yet been taken or released
        action_released.set(False)
        action_taken.set(False)

# Key release handlr for human player
def on_release_event(event):
    if(event.key in key_action_map):
        key_action = key_action_map[event.key]
        a = action.get()
        if(a==key_action):
            # If released action has already been taken set action to no-op immediately
            if(action_taken.get()):
                action.set(0)
            # Otherwise note that it has been released so that we can set back to no-op when it is taken
            else:
                action_released.set(True)

################################################################################################################
# play
#
# Allows user to the play the game and displays state and score via gui.
#
################################################################################################################
def play():
    if is_terminate.get() == True:
        gui.quit()

    # Get players actions selection, if that action has been released, set action back to no-op for next frame
    a = action.get()
    action_taken.set(True)
    if(action_released.get()):
        action.set(0)

    r, t = env.act(a)
    is_terminate.set(t)
    G.set(G.get()+r)
    gui.display_state(env.state())
    gui.set_message("Score: " + str(G.get()))
    gui.update(50, play)

# Hook up the key handler and initiate game play
gui.overwrite_key_handle(on_key_event, on_release_event)
gui.update(0, play)
gui.run()
print("Final Score: "+str(G.get()))
