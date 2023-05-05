################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
################################################################################################################
import numpy as np


#####################################################################################################################
# Constants
#
#####################################################################################################################
shot_cool_down = 5
enemy_move_interval = 12
enemy_shot_interval = 10


#####################################################################################################################
# Env
#
# The player controls a cannon at the bottom of the screen and can shoot bullets upward at a cluster of aliens above.
# The aliens move across the screen until one of them hits the edge, at which point they all move down and switch
# directions. The current alien direction is indicated by 2 channels (one for left and one for right) one of which is
# active at the location of each alien. A reward of +1 is given each time an alien is shot, and that alien is also
# removed. The aliens will also shoot bullets back at the player. When few aliens are left, alien speed will begin to
# increase. When only one alien is left, it will move at one cell per frame. When a wave of aliens is fully cleared a
# new one will spawn which moves at a slightly faster speed than the last. Termination occurs when an alien or bullet
# hits the player.
#
#####################################################################################################################
class Env:
    def __init__(self, ramping=True):
        self.channels ={
            'cannon':0,
            'alien':1,
            'alien_left':2,
            'alien_right':3,
            'friendly_bullet':4,
            'enemy_bullet':5
        }
        self.action_map = ['n','l','u','r','d','f']
        self.ramping = ramping
        self.random = np.random.RandomState()
        self.reset()

    # Update environment according to agent action
    def act(self, a):
        r = 0
        if(self.terminal):
            return r, self.terminal

        a = self.action_map[a]

        # Resolve player action
        if(a=='f' and self.shot_timer == 0):
            self.f_bullet_map[9,self.pos]=1
            self.shot_timer = shot_cool_down
        elif(a=='l'):
            self.pos = max(0, self.pos-1)
        elif(a=='r'):
            self.pos = min(9, self.pos+1)

        # Update Friendly Bullets
        self.f_bullet_map = np.roll(self.f_bullet_map, -1, axis=0)
        self.f_bullet_map[9,:] = 0

        # Update Enemy Bullets
        self.e_bullet_map = np.roll(self.e_bullet_map, 1, axis=0)
        self.e_bullet_map[0,:] = 0
        if(self.e_bullet_map[9,self.pos]):
            self.terminal = True

        # Update aliens
        if(self.alien_map[9,self.pos]):
            self.terminal = True
        if(self.alien_move_timer==0):
            self.alien_move_timer = min(np.count_nonzero(self.alien_map),self.enemy_move_interval)
            if((np.sum(self.alien_map[:,0])>0 and self.alien_dir<0) or (np.sum(self.alien_map[:,9])>0 and self.alien_dir>0)):
                self.alien_dir = -self.alien_dir
                if(np.sum(self.alien_map[9,:])>0):
                    self.terminal = True
                self.alien_map = np.roll(self.alien_map, 1, axis=0)
            else:
                self.alien_map = np.roll(self.alien_map, self.alien_dir, axis=1)
            if(self.alien_map[9,self.pos]):
                self.terminal = True
        if(self.alien_shot_timer==0):
            self.alien_shot_timer = enemy_shot_interval
            nearest_alien = self._nearest_alien(self.pos)
            self.e_bullet_map[nearest_alien[0], nearest_alien[1]] = 1

        kill_locations = np.logical_and(self.alien_map,self.alien_map==self.f_bullet_map)

        r+=np.sum(kill_locations)
        self.alien_map[kill_locations] = self.f_bullet_map[kill_locations] = 0


        # Update various timers
        self.shot_timer -= self.shot_timer>0
        self.alien_move_timer-=1
        self.alien_shot_timer-=1
        if(np.count_nonzero(self.alien_map)==0):
            if(self.enemy_move_interval>6 and self.ramping):
                self.enemy_move_interval-=1
                self.ramp_index+=1
            self.alien_map[0:4,2:8] = 1
        return r, self.terminal

    # Find the alien closest to player in manhattan distance, currently used to decide which alien shoots
    def _nearest_alien(self, pos):
        search_order = [i for i in range(10)]
        search_order.sort(key=lambda x: abs(x-pos))
        for i in search_order:
            if(np.sum(self.alien_map[:,i])>0):
                return [np.max(np.where(self.alien_map[:,i]==1)),i]
        return None

    # Query the current level of the difficulty ramp, could be used as additional input to agent for example
    def difficulty_ramp(self):
        return self.ramp_index

    # Process the game-state into the 10x10xn state provided to the agent and return
    def state(self):
        state = np.zeros((10,10,len(self.channels)),dtype=bool)
        state[9,self.pos,self.channels['cannon']] = 1
        state[:,:, self.channels['alien']] = self.alien_map
        if(self.alien_dir<0):
            state[:,:, self.channels['alien_left']] = self.alien_map
        else:
            state[:,:, self.channels['alien_right']] = self.alien_map
        state[:,:, self.channels['friendly_bullet']] = self.f_bullet_map
        state[:,:, self.channels['enemy_bullet']] = self.e_bullet_map
        return state

    # Reset to start state for new episode
    def reset(self):
        self.pos = 5
        self.f_bullet_map = np.zeros((10,10))
        self.e_bullet_map = np.zeros((10,10))
        self.alien_map = np.zeros((10,10))
        self.alien_map[0:4,2:8] = 1
        self.alien_dir = -1
        self.enemy_move_interval = enemy_move_interval
        self.alien_move_timer = self.enemy_move_interval
        self.alien_shot_timer = enemy_shot_interval
        self.ramp_index = 0
        self.shot_timer = 0
        self.terminal = False

    # Dimensionality of the game-state (10x10xn)
    def state_shape(self):
        return [10,10,len(self.channels)]

    # Subset of actions that actually have a unique impact in this environment
    def minimal_action_set(self):
        minimal_actions = ['n','l','r','f']
        return [self.action_map.index(x) for x in minimal_actions]
