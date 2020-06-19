##############################################################################################################################
#Policy iteration has been implemented on a preexisting third party environment called "grid world"                          #
#It can be found at https://github.com/MJeremy2017/Reinforcement-Learning-Implementation/blob/master/GridWorld/gridWorld_Q.py#
#This algorithm shows the difference between q-learning which was originally implemented and policy iteration                #
##############################################################################################################################

import numpy as np

# global variables
BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (2, 3)
LOSE_STATE = (1, 1)
START = (0, 0)
DETERMINISTIC = True
discount_factor=0.9 # policy evaluation is not converging for discount factor =1


                  ####### Choose Algo : [0 for q-learning], [1 for policy iteration] #######
algo=1


class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[1, 1] = -1              
        self.state = state                  
        self.isEnd = False                  
        self.determine = DETERMINISTIC      # No noise

    
    def giveReward(self): #Reward protocol 
        if self.state == WIN_STATE:
            return 1
        elif self.state == LOSE_STATE:

            return -1
        else:
            return -0.1

    def giveNxtReward(self,nxtpos): # Useful for policy iteration algo
        if nxtpos == WIN_STATE:
            return 10
        elif nxtpos == LOSE_STATE:
            return -10
        else:
            return -0.1

    def isEndFunc(self):
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.isEnd = True
 
    def nxtPosition(self, action):

        if self.determine:
            if action == 'u':
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == 'd':
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == 'l':
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
 
            if (nxtState[0] >= 0) and (nxtState[0] <= BOARD_ROWS-1):       
                if (nxtState[1] >= 0) and (nxtState[1] <= BOARD_COLS-1):
                    #if nxtState != (1, 1): Don't bump into wall
                    return nxtState
            return self.state

class Agent:

    def __init__(self):
        self.states = []
        self.actions = ['u', 'd', 'l', 'r']
        self.State = State()
        self.lr = 0.2
        self.exp_rate = 0.3

        # initial state reward
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.state_values[(i, j)] = 0  

    def reset(self):
        self.states = []
        self.State = State()

    def chooseAction(self):

        mx_nxt_reward = 0 # to store best next reward among all possible actions
        action = ""
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:

            for a in self.actions:
                # if the action is deterministic
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward 
        return action 


    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)


                                            # policy iteration methods #
    def policy_evaluation(self,theta,V,V_old):
        print("Policy evaluation")
        compteur_iteration = 0
        while True:
             delta=0
             for x in range(0, BOARD_ROWS):
                 for y in range(0, BOARD_COLS):
                    V_old[(x,y)]=V[(x,y)] # pour ne pas avoir le meme objet V_old=V 
                    print(V_old)
             for x in range(0, BOARD_ROWS):
                 for y in range(0, BOARD_COLS):
                     self.State.state = (x, y)
                     nxtState = self.State.nxtPosition(policy[(x, y)])
                     if (x,y)== WIN_STATE or (x,y)== LOSE_STATE:
                         continue
                     nxtreward = self.State.giveNxtReward(nxtState)
                     reward=self.State.giveReward()
                     v = V[(x, y)]
                     V[(x, y)] = reward + discount_factor * V_old[nxtState]
                     delta = max(delta, abs(v - V[(x, y)]))
             compteur_iteration+=1
             if delta<theta:
                 print(compteur_iteration)
                 return V
                 break

    def policy_improvement(self,V):
        print("Policy improvement")
        policy_stable = True

        for x in range(0, BOARD_ROWS):
            for y in range(0, BOARD_COLS):
                self.State.state = (x, y)
                a = policy[(x, y)]
                v = V[self.State.nxtPosition(policy[(x, y)])]
                a_star = a
                v_star = v
                for action in ['u', 'd', 'l', 'r']:
                    v_prime = self.State.giveReward() + discount_factor * V[self.State.nxtPosition(action)]
                    if v_prime > v_star:
                        a_star = action
                        v_star = v_prime
                        policy_stable = False
                    policy[(x, y)] = a_star
        return policy_stable


    def play(self, rounds=10):
            i = 0
            while i < rounds:
                if self.State.isEnd:
                    reward = self.State.giveReward()
                    self.state_values[self.State.state] = reward  
                    print("Game End Reward", reward)
                    for s in reversed(self.states):
                        reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                        self.state_values[s] = round(reward, 3)
                    self.reset()
                    i += 1
                else:
                    action = self.chooseAction()

                    self.states.append(self.State.nxtPosition(action))
                    print("current position {} action {}".format(self.State.state, action))
                    self.State = self.takeAction(action)
                    self.State.isEndFunc()
                    print("nxt state", self.State.state)
                    print("---------------------")

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')

    def showValueFunction(self,V):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(round(V[(i, j)], 3)) + ' | '
            print(out)
        print('----------------------------------')

                                            
if __name__ == "__main__":

    ag = Agent()
    if algo==0:  #Q-learing
         ag.play(100)
         print(ag.showValues())

    elif algo==1: #Policy-iteration
        policy = np.empty(shape=(BOARD_ROWS, BOARD_COLS),dtype=str)
        for x in range(0, BOARD_ROWS):
            for y in range(0, BOARD_COLS):
                policy[(x, y)] = np.random.choice(ag.actions)
        print(policy)
        V = np.zeros(shape=(BOARD_ROWS, BOARD_COLS))
        V[(1, 1)] = -10
        V[(2, 3)] = 10
        V_old = np.zeros(shape=(BOARD_ROWS, BOARD_COLS))

        while True:
            V=ag.policy_evaluation(0.01,V,V_old)
            policy_stable=ag.policy_improvement(V)
            print(policy)
            if(policy_stable==True):
                print(policy)
                ag.showValueFunction(V)
                break