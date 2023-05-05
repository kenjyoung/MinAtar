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
ramp_interval = 100
max_oxygen = 200
init_spawn_speed = 20
diver_spawn_speed = 30
init_move_interval = 5
shot_cool_down = 5
enemy_shot_interval = 10
enemy_move_interval = 5
diver_move_interval = 5


#####################################################################################################################
# Env
#
# The player controls a submarine consisting of two cells, front and back, to allow direction to be determined. The
# player can also fire bullets from the front of the submarine. Enemies consist of submarines and fish, distinguished
# by the fact that submarines shoot bullets and fish do not. A reward of +1 is given each time an enemy is struck by
# one of the player's bullets, at which point the enemy is also removed. There are also divers which the player can
# move onto to pick up, doing so increments a bar indicated by another channel along the bottom of the screen. The
# player also has a limited supply of oxygen indicated by another bar in another channel. Oxygen degrades over time,
# and is replenished whenever the player moves to the top of the screen as long as the player has at least one rescued
# diver on board. The player can carry a maximum of 6 divers. When surfacing with less than 6, one diver is removed.
# When surfacing with 6, all divers are removed and a reward is given for each active cell in the oxygen bar. Each
# time the player surfaces the difficulty is increased by increasing the spawn rate and movement speed of enemies.
# Termination occurs when the player is hit by an enemy fish, sub or bullet; or when oxygen reached 0; or when the
# player attempts to surface with no rescued divers. Enemy and diver directions are indicated by a trail channel
# active in their previous location to reduce partial observability.
#
#####################################################################################################################
class Env:
    def __init__(self, ramping=True):
        self.channels ={
            'sub_front':0,
            'sub_back':1,
            'friendly_bullet':2,
            'trail':3,
            'enemy_bullet':4,
            'enemy_fish':5,
            'enemy_sub':6,
            'oxygen_guage':7,
            'diver_guage':8,
            'diver':9
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

        # Spawn enemy if timer is up
        if(self.e_spawn_timer==0):
            self._spawn_enemy()
            self.e_spawn_timer = self.e_spawn_speed

        if(self.d_spawn_timer==0):
            self._spawn_diver()
            self.d_spawn_timer = diver_spawn_speed

        # Resolve player action
        if(a=='f' and self.shot_timer == 0):
            self.f_bullets+=[[self.sub_x, self.sub_y, self.sub_or]]
            self.shot_timer = shot_cool_down
        elif(a=='l'):
            self.sub_x = max(0, self.sub_x-1)
            self.sub_or = False
        elif(a=='r'):
            self.sub_x = min(9, self.sub_x+1)
            self.sub_or = True
        elif(a=='u'):
            self.sub_y = max(0, self.sub_y-1)
        elif(a=='d'):
            self.sub_y = min(8, self.sub_y+1)

        # Update friendly Bullets
        for bullet in reversed(self.f_bullets):
            bullet[0]+=1 if bullet[2] else -1
            if(bullet[0]<0 or bullet[0]>9):
                self.f_bullets.remove(bullet)
            else:
                removed = False
                for x in self.e_fish:
                    if(bullet[0:2]==x[0:2]):
                        self.e_fish.remove(x)
                        self.f_bullets.remove(bullet)
                        r+=1
                        removed = True
                        break
                if(not removed):
                    for x in self.e_subs:
                        if(bullet[0:2]==x[0:2]):
                            self.e_subs.remove(x)
                            self.f_bullets.remove(bullet)
                            r+=1
                            break

        # Update divers
        for diver in reversed(self.divers):
            if(diver[0:2]==[self.sub_x,self.sub_y] and self.diver_count<6):
                self.divers.remove(diver)
                self.diver_count+=1
            else:
                if(diver[3]==0):
                    diver[3]=diver_move_interval
                    diver[0]+=1 if diver[2] else -1
                    if(diver[0]<0 or diver[0]>9):
                        self.divers.remove(diver)
                    elif(diver[0:2]==[self.sub_x,self.sub_y] and self.diver_count<6):
                        self.divers.remove(diver)
                        self.diver_count+=1
                else:
                    diver[3]-=1

        # Update enemy subs
        for sub in reversed(self.e_subs):
            if(sub[0:2]==[self.sub_x,self.sub_y]):
                self.terminal = True
            if(sub[3]==0):
                sub[3]=self.move_speed
                sub[0]+=1 if sub[2] else -1
                if(sub[0]<0 or sub[0]>9):
                    self.e_subs.remove(sub)
                elif(sub[0:2]==[self.sub_x,self.sub_y]):
                    self.terminal = True
                else:
                    for x in self.f_bullets:
                        if(sub[0:2]==x[0:2]):
                            self.e_subs.remove(sub)
                            self.f_bullets.remove(x)
                            r+=1
                            break
            else:
                sub[3]-=1
            if(sub[4]==0):
                sub[4]=enemy_shot_interval
                self.e_bullets+=[[sub[0] if sub[2] else sub[0], sub[1], sub[2]]]
            else:
                sub[4]-=1

        # Update enemy bullets
        for bullet in reversed(self.e_bullets):
            if(bullet[0:2]==[self.sub_x,self.sub_y]):
                self.terminal = True
            bullet[0]+=1 if bullet[2] else -1
            if(bullet[0]<0 or bullet[0]>9):
                self.e_bullets.remove(bullet)
            else:
                if(bullet[0:2]==[self.sub_x,self.sub_y]):
                    self.terminal = True

        # Update enemy fish
        for fish in reversed(self.e_fish):
            if(fish[0:2]==[self.sub_x,self.sub_y]):
                self.terminal = True
            if(fish[3]==0):
                fish[3]=self.move_speed
                fish[0]+=1 if fish[2] else -1
                if(fish[0]<0 or fish[0]>9):
                    self.e_fish.remove(fish)
                elif(fish[0:2]==[self.sub_x,self.sub_y]):
                    self.terminal = True
                else:
                    for x in self.f_bullets:
                        if(fish[0:2]==x[0:2]):
                            self.e_fish.remove(fish)
                            self.f_bullets.remove(x)
                            r+=1
                            break
            else:
                fish[3]-=1

        # Update various timers
        self.e_spawn_timer -= self.e_spawn_timer>0
        self.d_spawn_timer -= self.d_spawn_timer>0
        self.shot_timer -= self.shot_timer>0
        if(self.oxygen<=0):
            self.terminal = True
        if(self.sub_y>0):
            self.oxygen-=1
            self.surface = False
        else:
            if(not self.surface):
                if(self.diver_count == 0):
                    self.terminal = True
                else:
                    r+=self._surface()
        return r, self.terminal

    # Called when player hits surface (top row) if they have no divers, this ends the game,
    # if they have 6 divers this gives reward proportional to the remaining oxygen and restores full oxygen
    # otherwise this reduces the number of divers and restores full oxygen
    def _surface(self):
        self.surface = True
        if(self.diver_count == 6):
            self.diver_count = 0
            r = self.oxygen*10//max_oxygen
        else:
            r = 0
        self.oxygen = max_oxygen
        self.diver_count -= 1
        if self.ramping and (self.e_spawn_speed>1 or self.move_speed>2):
            if(self.move_speed>2 and self.ramp_index%2):
                    self.move_speed-=1
            if(self.e_spawn_speed>1):
                    self.e_spawn_speed-=1
            self.ramp_index+=1
        return r

    # Spawn an enemy fish or submarine in random row and random direction,
    # if the resulting row and direction would lead to a collision, do nothing instead
    def _spawn_enemy(self):
        lr = self.random.rand() < 1/2
        is_sub = self.random.rand() < 1/3
        x = 0 if lr else 9
        y = self.random.randint(low=1, high=9)

        # Do not spawn in same row an opposite direction as existing
        if(any([z[1]==y and z[2]!=lr for z in self.e_subs+self.e_fish])):
            return
        if(is_sub):
            self.e_subs+=[[x,y,lr,self.move_speed,enemy_shot_interval]]
        else:
            self.e_fish+=[[x,y,lr,self.move_speed]]

    # Spawn a diver in random row with random direction
    def _spawn_diver(self):
        lr = self.random.rand() < 1/2
        x = 0 if lr else 9
        y = self.random.randint(low=1, high=9)
        self.divers+=[[x,y,lr,diver_move_interval]]

    # Query the current level of the difficulty ramp, could be used as additional input to agent for example
    def difficulty_ramp(self):
        return self.ramp_index

    # Process the game-state into the 10x10xn state provided to the agent and return
    def state(self):
        state = np.zeros((10,10,len(self.channels)),dtype=bool)
        state[self.sub_y,self.sub_x,self.channels['sub_front']] = 1
        back_x = self.sub_x-1 if self.sub_or else self.sub_x+1
        state[self.sub_y,back_x,self.channels['sub_back']] = 1
        state[9,0:max(0,self.oxygen)*10//max_oxygen, self.channels['oxygen_guage']] = 1
        state[9,9-self.diver_count:9, self.channels['diver_guage']] = 1
        for bullet in self.f_bullets:
            state[bullet[1],bullet[0], self.channels['friendly_bullet']] = 1
        for bullet in self.e_bullets:
            state[bullet[1],bullet[0], self.channels['enemy_bullet']] = 1
        for fish in self.e_fish:
            state[fish[1],fish[0], self.channels['enemy_fish']] = 1
            back_x = fish[0]-1 if fish[2] else fish[0]+1
            if(back_x>=0 and back_x<=9):
                state[fish[1],back_x, self.channels['trail']] = 1
        for sub in self.e_subs:
            state[sub[1],sub[0], self.channels['enemy_sub']] = 1
            back_x = sub[0]-1 if sub[2] else sub[0]+1
            if(back_x>=0 and back_x<=9):
                state[sub[1],back_x, self.channels['trail']] = 1
        for diver in self.divers:
            state[diver[1],diver[0], self.channels['diver']] = 1
            back_x = diver[0]-1 if diver[2] else diver[0]+1
            if(back_x>=0 and back_x<=9):
                state[diver[1],back_x, self.channels['trail']] = 1

        return state

    # Reset to start state for new episode
    def reset(self):
        self.oxygen = max_oxygen
        self.diver_count = 0
        self.sub_x = 5
        self.sub_y = 0
        # 0=left, 1=right
        self.sub_or = False
        self.f_bullets = []
        self.e_bullets = []
        self.e_fish = []
        self.e_subs = []
        self.divers = []
        self.e_spawn_speed = init_spawn_speed
        self.e_spawn_timer = self.e_spawn_speed
        self.d_spawn_timer = diver_spawn_speed
        self.move_speed = init_move_interval
        self.ramp_index = 0
        self.shot_timer = 0
        self.surface = True
        self.terminal = False

    # Dimensionality of the game-state (10x10xn)
    def state_shape(self):
        return [10,10,len(self.channels)]

    # Subset of actions that actually have a unique impact in this environment
    def minimal_action_set(self):
        minimal_actions = ['n','l','u','r','d','f']
        return [self.action_map.index(x) for x in minimal_actions]
