import numpy as np
import random
import pandas as pd

###############################################################
#
# Created on 03/01/2024 by Nicola Rossi (@nicolaxrossi)
# Last modified 06/01/2024
#
###############################################################

class Environment:

    GAMMA = 1

    ACTIONS = {'up':(-1,0),
               'down':(1,0),
               'left':(0,-1),
               'right':(0,1)}

    def __init__(self):

        # randomly initialize the agent's position
        self.position = (random.randint(1, 4), random.randint(1, 8))
        #self.position = (4,1)

        # if the agent's initial position is (2,2) or (2,7) (where the tools are located), then tool is immediately set to True,
        # assuming that if the agent spwans on the tool it immediately collects it
        if self.position == (2,1) or self.position == (2,7):
            self.tool = True

        else:
            self.tool = False


        # then, datasets containing part of dynamics (for legal actions) are opened as pandas dataframes
        self.up_df = pd.read_csv('up.csv')
        self.down_df = pd.read_csv('down.csv')
        self.left_df = pd.read_csv('left.csv')
        self.right_df = pd.read_csv('right.csv')

    def get_return(self):
        """ given a state ((i,j), tool) it returns the associated return (check the documentation for details) """

        # if the agent arrives in (2,1) and does not have the tool (it is 'picking' the tool 'on arrival' on (2,1)), then the reward is 0
        if self.position == (2,1) and not self.tool:
            self.tool = True
            return 0

        # same as before, for tool in (2,7)
        if self.position == (2,7) and not self.tool:
            self.tool = True
            return 0

        # if the agent has collected the tool and it's now in position (1,7) it has accomplished its goal
        if self.tool and self.position == (1,7):
            return 0
        
        # if the agent has collected the tool but it hasn't yet reached the oven, then the return is -1
        if self.tool and self.position != (1,7):
            return -1
        
        # if all the previous cases are false, then the agent has to collect the tool; so the reward is -1
        return -1

    
    def apply(self, action):

        """ it's assumed that
         1) the input action is in {'up', 'down', 'left', 'right'}
         2) the input action is legal for the current state 
        
        This function return the state obtained by applying the input action to the current state and the associated return """


        # code to manage the gates
        # gate in (3,4)
        if self.position == (3,4) and action == 'right':
            self.position = (2,8)

            return ((self.position, self.tool), self.get_return())
        
        # gate in (2,8)
        if self.position == (2,8) and action == 'right':
            self.position = (3,4)

            return ((self.position, self.tool), self.get_return())
        
        # firstly, the action is applied to the current state
        # 1) the new position is calculated and the current position is updated
        direction = self.ACTIONS[action]
        new_position = (self.position[0] + direction[0], self.position[1] + direction[1])
        self.position = new_position

        # 2) it's checked whether the tool variable must be updated (set to True)
        #if self.position == (2,1) or self.position == (2,7):
        #    self.tool = True

        # 3) the return is calculated
        ret = self.get_return()

        return ((self.position, self.tool), ret)

    
    def get_curr_state(self):
        """ return the current state, a couple made by:
         - current position (i,j)
         - the flag indicating if the agent has taken the tool """
        return (self.position, self.tool)
    
    def enumerate_state(self, pos):

        """ simple function to associate to each state a natural number. 
         Since there are 64 possible states, numbers between 0 and 63 are mapped to states """
        
        i = pos[0]
        j = pos[1]

        return 8*(i-1) + (j-1)

    def inverse(self, index):

        """ a simple function that computes the inverse of enumerate_state function """
        i = 0

        if 0 <= index <= 7:
            i = 1
        elif 8 <= index <= 15:
            i = 2
        elif 16 <= index <= 23:
            i = 3
        elif 24 <= index <= 31:
            i = 4
        
        j = index - 8*(i-1) + 1

        return (i,j)
    
    def get_legal_actions(self, pos, tool):
        """ return the list of legal actions from the current state: in this case the transition tensor is used """

        # if the agent has already taken the tool and it's currently in the oven's cell, then the episode is ended, so None is returned
        # notice that this is the only case in which for a state no legal action exists!
        if tool and pos == (1,7):
            return None
        
        # the current state is mapped to a natural number (an index) in order to access the relative matrix, action by action
        index = self.enumerate_state(pos)

        legal_actions = []

        # UP
        up_vector = list(self.up_df.iloc[index, 1:])
        for entry in up_vector:
            if entry == 1:
                legal_actions.append('up')
                break

        # DOWN
        down_vector = list(self.down_df.iloc[index, 1:])
        for entry in down_vector:
            if entry == 1:
                legal_actions.append('down')
                break

        # LEFT
        left_vector = list(self.left_df.iloc[index, 1:])
        for entry in left_vector:
            if entry == 1:
                legal_actions.append('left')
                break

        # RIGHT
        right_vector = list(self.right_df.iloc[index, 1:])
        for entry in right_vector:
            if entry == 1:
                legal_actions.append('right')
                break
        
        return legal_actions

    def reset(self):

        self.position = (random.randint(1, 4), random.randint(1, 8))

        if self.position == (2,1) or self.position == (2,7):
            self.tool = True

        else:
            self.tool = False

    def rep(self):
        rep = [['.' for j in range(0,9)] for i in range(0,4)]

        rep[0][4] = '|'
        rep[1][4] = '|'
        rep[2][4] = '|'
        rep[3][4] = '|'

        rep[1][0] = 't'
        rep[1][7] = 't'
        rep[0][7] = 'o'
        rep[2][3] = 'G'
        rep[1][8] = 'G'

        if self.position[1] >= 4:
            rep[self.position[0]-1][self.position[1]] = '*'
        else:
            rep[self.position[0]-1][self.position[1]-1] = '*'

        rep_str = ''
        for i in range(0,4):
            rep_str += str(rep[i]) + '\n' 
        
        print(rep_str)


if __name__ == '__main__':

    env = Environment()

    env.rep()

    s, r = env.apply('up')
    print(s)
    print(r)
    env.rep()

    s, r = env.apply('up')
    print(s)
    print(r)
    env.rep()

    s, r = env.apply('right')
    print(s)
    print(r)
    env.rep()

    s, r = env.apply('down')
    s, r = env.apply('right')
    s, r = env.apply('right')
    s, r = env.apply('right')
    env.rep()

    s, r = env.apply('up')
    s, r = env.apply('left')
    print(s)
    print(r)
    env.rep()

    print(env.get_legal_actions((3,6), False))