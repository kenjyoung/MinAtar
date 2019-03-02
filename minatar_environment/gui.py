################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
#                                                                                                              #
################################################################################################################

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as colors
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
import tkinter as Tk

from os import system
from platform import system as platform


################################################################################################################
# class GUI
#
# Host the visualization of a game play and handle keyboard input (e.g. to allow human play or display agent 
# player).
#
################################################################################################################
class GUI:
    def __init__(self, env_name, n_channels):
        self.n_channels = n_channels

        # The seaborn color_palette cubhelix is used to assign visually distinct colors to each channel for the env
        self.cmap = sns.color_palette("cubehelix", self.n_channels)
        self.cmap.insert(0, (0, 0, 0))
        self.cmap = colors.ListedColormap(self.cmap)
        bounds = [i for i in range(self.n_channels + 2)]
        self.norm = colors.BoundaryNorm(bounds, self.n_channels + 1)

        self.root = Tk.Tk()
        self.root.title(env_name)
        self.root.config(background='white')

        self.root.attributes("-topmost", True)
        if platform() == 'Darwin':  # How Mac OS X is identified by Python
            system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')
        self.root.focus_force()

        self.text_message = Tk.StringVar()
        self.label = Tk.Label(self.root, textvariable=self.text_message)

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.key_press_handler = self.canvas.mpl_connect('key_press_event', self.on_key_event)
        self.key_release_handler = self.canvas.mpl_connect('key_press_event', lambda x: None)

    # Set the message for the label on screen
    def set_message(self, str):
        self.text_message.set(str)
        self.label.pack()

    # Show the current frame
    def display_state(self, state):
        self.ax.cla()
        numerical_state = np.amax(state * np.reshape(np.arange(self.n_channels) + 1, (1, 1, -1)), 2) + 0.5
        self.ax.imshow(numerical_state, cmap=self.cmap, norm=self.norm, interpolation='none')
        self.canvas.draw()

    # Allow user to handle their own keyboard input
    def overwrite_key_handle(self, key_press_handler, key_release_handler=None):
        self.canvas.mpl_disconnect(self.key_press_handler)
        self.key_press_handler = self.canvas.mpl_connect('key_press_event', key_press_handler)
        if(key_release_handler is not None):
            self.canvas.mpl_disconnect(self.key_release_handler)
            self.key_release_handler = self.canvas.mpl_connect('key_release_event', key_release_handler)


    # Default key handler
    def on_key_event(self, event):
        if event.key == "q":
            self.quit()

    # Quit the GUI
    def quit(self):
        self.root.quit()

    # After millisecond, calls function func
    def update(self, millisecond, func):
        self.root.after(millisecond, func)

    # Start the GUI
    def run(self):
        self.root.mainloop()
