######################################################################################
#This notebook implements a Deep Q Network to learn how to play the game MountainCar.#
#The state of the game is here a tuple of two numbers (position and velocity)        #
#returned by the environment.                                                        #
######################################################################################

import gym
import numpy as np
import torch
import random
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):  # nn model 2 states --> 64layer ---> 64layer ----> 3 Q-action pairs
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 512, bias=False)  # 2d state space
        #self.fc2 = nn.Linear(64, 64, bias=False)
        self.fc3 = nn.Linear(512, action_dim, bias=True)  # one output per action
        self.device = torch.device("cpu")

    def forward(self, x):  # forward function
        state = torch.Tensor(x).to(self.device)
        x = F.relu(self.fc1(state))
        #x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class Agent:
    def __init__(self, state_dim, action_dim):
        self.network = DQN(state_dim, action_dim).to(device)  # creates network from nn model
        self.target_network = copy.deepcopy(self.network).to(device)  # create target_network from network
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.005)  # optimizer, parameters
        self.memory = []
        self.states_memory = []
        self.actions_memory = []
        self.rewards_memory = []
        self.next_states_memory = []
        self.dones_memory = []

    def update(self, batch, update_dqn_target):
        states, actions, rewards, next_states, dones = zip(*batch)  # from random batch ( from buffer )
        self.states_memory = torch.from_numpy(np.array(states)).float().to(device)  # collect states
        self.actions_memory = torch.from_numpy(np.array(actions)).to(device).unsqueeze(1)  # collect actions
        self.rewards_memory = torch.from_numpy(np.array(rewards)).float().to(device).unsqueeze(1)  # collect rewards
        self.next_states_memory = torch.from_numpy(np.array(next_states)).float().to(device)  # etc...
        self.dones_memory = torch.from_numpy(np.array(dones)).to(device).unsqueeze(1)
        if not update_dqn_target % UPDATE_TARGET:
            agent.target_network = copy.deepcopy(agent.network)

    def train(self):
        target = self.rewards_memory + (gamma * self.target_network(self.next_states_memory).detach().max(1)[0].unsqueeze(1)) * (~self.dones_memory)  # q-target   


        Q_current = self.network(self.states_memory).gather(-1, self.actions_memory)  # Q_current,  gather will index the rows of the q-values
        self.optimizer.zero_grad()  # optimizer
        loss = F.smooth_l1_loss(target, Q_current)  # loss
        loss.backward()  # backward
        self.optimizer.step()  # step optimizer -----> improves network

    def act(self, env, state, eps):
        if random.random() < eps:
            return env.action_space.sample()
        state = torch.tensor(state).to(device).float()
        Q_values = self.network(state.unsqueeze(0)).detach()
        return np.argmax(Q_values.cpu().data.numpy())  # choose action from state ---> network

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, scores, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
       for line in lines:
           plt.axvline(x=line)

    plt.savefig(filename)


if __name__ == '__main__':
    BATCH_SIZE = 64
    UPDATE_TARGET = 128
    gamma = 0.98
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    env = gym.make('MountainCar-v0')
    env.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    agent = Agent(state_dim=2, action_dim=3)
    episodes = 3000
    eps = 1
    eps_coeff = 0.995
    dqn_updates = 0
    rewards = []
    epsh= []
    mean_rh = []
    cpt = 0

    step = -1
    for i in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            step += 1
            action = agent.act(env, state, eps)
            next_state, reward, done, _ = env.step(action)
            #next_state = relu_compat(next_state)
            total_reward += reward
            if step < 20000:
                agent.memory.append(None)
            agent.memory[step % 20000] = (state, action, reward, next_state, done)
            
            
            if step >= 10000: #and total_reward<-120:
                sample = random.sample(agent.memory, BATCH_SIZE)
                agent.update(sample, step)
                agent.train()


            state = next_state
            if state[0] >= env.goal_position:
            
                cpt += 1
                break
        eps *= eps_coeff

        # show every 5 ep
        rewards.append(total_reward)
        epsh.append(eps)
        mean_r = np.mean(rewards)
        mean_rh.append(mean_r)
        max_r = np.max(rewards)
        if i % 5 == 0:
            print(f'\repisode {i}, eps = {eps}, mean = {mean_r}, max = {max_r}, accuracy = {cpt}/{i}')

x = [i+1 for i in range(episodes)]
filename = "plot.png"
plotLearning(x, rewards, epsh, filename)
