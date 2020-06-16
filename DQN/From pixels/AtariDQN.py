import gym
import numpy as np
import torch
import random
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
from os.path import exists
from wrappers import *


load = False

env_id = "PongNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):  
        super().__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, action_dim)
        self.device = 'cpu'

    def forward (self, x):
        state = x.to(self.device).float()/255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))   #flattening
        actions = self.fc2(x)
        return actions



class Agent:
    def __init__(self, state_dim, action_dim):
        self.network = DQN(state_dim, action_dim).to(device)  # creates network from nn model
        self.target_network = copy.deepcopy(self.network).to(device)  # create target_network from network
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.00025)  # optimizer, parameters, lr very low
        self.memory = []
        self.states_memory = []
        self.actions_memory = []
        self.rewards_memory = []
        self.next_states_memory = []
        self.dones_memory = []

    def update(self, batch, update_dqn_target):
        states, actions, rewards, next_states, dones = zip(*batch)  # from random batch ( from buffer )
        self.states_memory = torch.from_numpy(np.array(states)).float().to(device)  # collect states
        self.actions_memory = torch.from_numpy(np.array(actions)).to(device).unsqueeze(1).type(torch.int64)  # collect actions
        self.rewards_memory = torch.from_numpy(np.array(rewards)).float().to(device).unsqueeze(1)  # collect rewards
        self.next_states_memory = torch.from_numpy(np.array(next_states)).float().to(device)  # etc...
        self.dones_memory = torch.from_numpy(np.array(dones)).to(device).unsqueeze(1)
        if not update_dqn_target % UPDATE_TARGET:
            agent.target_network = copy.deepcopy(agent.network)

    def train(self):
        with torch.no_grad():
          argmax = self.network(self.next_states_memory).detach().max(1)[1].unsqueeze(1)  # action max from next_states from batch above
          target = self.rewards_memory + (gamma * self.target_network(self.next_states_memory).detach().gather(1, argmax)) * (~self.dones_memory)  # q-target

        Q_current = self.network(self.states_memory).gather(-1, self.actions_memory)  # Q_current
        self.optimizer.zero_grad()  # optimizer
        loss = F.mse_loss(target, Q_current)  # loss
        loss.backward()  # backward
        self.optimizer.step()  # step optimizer -----> improves network

    def act(self, env, state, eps):
        if random.random() < eps:
            return env.action_space.sample()
        state = torch.tensor(state).to(device).float()
        with torch.no_grad():
            Q_values = self.network(state.unsqueeze(0))
        return np.argmax(Q_values.cpu().data.numpy())  # choose action from state ---> network

      
    def save(self, episode, mean_reward):
        file_name = "trainedmodels/" + "pong" + ".ptm"
        agent_state = { "episode" : episode,
                        "network" : self.network.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "mean_reward": mean_reward};
        torch.save(agent_state, file_name)
        print("Agent's state saved to ", file_name)

    def load(self):
        file_name = "trainedmodels/" + "pong" + ".ptm"
        checkpoint = torch.load(file_name)
        self.network.load_state_dict(checkpoint["network"])
        self.network.to(device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode = checkpoint["episode"]
        self.mean_reward = checkpoint["mean_reward"]
        print("Loaded network model state from", file_name,
              "stopped at episode:", self.episode,
              " which fetched a mean reward of:", self.mean_reward)


if __name__ == '__main__':
    BATCH_SIZE = 32
    UPDATE_TARGET = 1000
    gamma = 0.99
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    env.reset()
    env.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    agent = Agent(env.observation_space.shape , env.action_space.n)
    episodes = 50000
    eps = 1
    eps_coeff = 0.99
    dqn_updates = 0
    rewards = []
    epsh= []
    mean_rh = []
    moy = 0
    true_rew = 0
    debut = 1
    if load == True :
      agent.load()
      debut = np.uint64(agent.episode)

    step = -1
    for i in range(debut, episodes + 1):
        state = env.reset()
        '''plt.figure()
        plt.imshow(state,
        interpolation='none', cmap='gray')
        plt.title('Example extracted screen')
        plt.colorbar()
        plt.show()'''
        done = False
        total_reward = 0
        
        while not done:
            #env.render()
            step += 1
            action = agent.act(env, state, eps)
            #action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            '''next_state = preprocess(next_state)
            plt.figure()
            plt.imshow(next_state,
            interpolation='none', cmap = 'gray')
            plt.title('Example extracted screen')
            plt.colorbar()
            plt.show()'''
            total_reward += reward
            eps = max(0.02, 1 - step / (3*10**5))

            if step < 75000:
                agent.memory.append(None)
            agent.memory[step % 75000] = (state, action, reward, next_state, done)

            if step >= 10000:
                sample = random.sample(agent.memory, BATCH_SIZE)
                agent.update(sample, step)
                agent.train()


            state = next_state

        moy += total_reward

        if i % 10 == 0 :
            moy10 = moy/10
            print(f'\rmoyenne10ep {moy10},episode {i},epsilon = {eps}, step = {step}')
            #if i % 100 == 0 :
              #agent.save(i, moy10)
            moy = 0