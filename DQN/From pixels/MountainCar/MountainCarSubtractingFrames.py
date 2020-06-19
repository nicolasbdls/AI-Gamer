#######################################################################################################
#This algorithm improves the agent's policy in MountainCar with the screen as the input with a DQN    #
#The method applied here is subtracting two consecutive preprocessed frames to have some information  # 
#about both the position and the velocity of the car                                                  #
#The agent begins winning after 1k episode but a clear convergence of the results couldn't be observed#
#######################################################################################################


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


class DQN(nn.Module):

    def __init__(self, state_dim, action_dim,h = 40, w = 60):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=3, stride=2, bias = False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, bias = False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, bias = False)

        self.device = 'cuda'

        # Number of Linear input connections depends on output of conv2d layers --> compute it
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(768, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        actions = self.fc1(x.view(x.size(0), -1))
        return actions



class Agent:
    def __init__(self, state_dim, action_dim):
        self.network = DQN(state_dim, action_dim).to(device)  # creates network from nn model
        self.target_network = copy.deepcopy(self.network).to(device)  # create target_network from network
        self.optimizer = optim.Adam(self.network.parameters(), lr=5e-4)  # optimizer, parameters, lr very low
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
          argmax = self.network(self.next_states_memory).detach().max(1)[1].unsqueeze(1)  # implements double Q-learning
          target = self.rewards_memory + (gamma * self.target_network(self.next_states_memory).detach().gather(-1, argmax)) * (~self.dones_memory)  # q-target
        
        Q_current = self.network(self.states_memory).gather(1, self.actions_memory)  # Q_current
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

def rgbtogray(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::10, ::10]

def get_screen():
	screen = env.render(mode = 'rgb_array')  #get the values of the pixels
	processed_state = np.uint8(np.reshape(np.uint8(rgbtogray(downsample(screen))), (1, 40, 60)))
	processed_state[processed_state != 128] = 0  #black background
	processed_state[processed_state == 128] = 1  #white wheels
	return processed_state




if __name__ == '__main__':
    BATCH_SIZE = 64
    UPDATE_TARGET = 100
    gamma = 0.98
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    env = gym.make('MountainCar-v0')
    env.reset()
    env.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    agent = Agent(state_dim= 1 , action_dim=env.action_space.n)  #single channel image
    episodes = 5000
    eps = 1
    eps_coeff = 0.99
    dqn_updates = 0
    rewards = []
    epsh= []
    mean_rh = []
    moy = 0
    step = -1
    for i in range(episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen()
        '''plt.figure()                                 #uncomment to visualize the processed image
        plt.imshow(last_screen.reshape(40,60),
        interpolation='none', cmap='gray')
        plt.title('Example extracted screen')
        plt.colorbar()
        plt.show()
        print(last_screen)'''
        current_screen = get_screen()
        state = current_screen - last_screen


        done = False
        total_reward = 0
        while not done:
            # Select and perform an action
            step += 1
            action = agent.act(env, state, eps)
            _, reward, done, _ = env.step(action)
            total_reward += reward

            # Observe new state
            if step% 4 == 0:
                last_screen = current_screen
                current_screen = get_screen()

                if not done:
                    next_state = current_screen + -1*last_screen   # subtracting next state with state
                    if np.amax(next_state) == 0 :
                        next_state = current_screen        
                else:
                    next_state = current_screen - current_screen

                plt.figure()
                plt.imshow(current_screen,
                interpolation='none')
                plt.title('Example extracted screen')
                plt.show()

            if step < 10000:
                agent.memory.append(None)
                agent.memory[step % 10000] = (state, action, reward, next_state, done)

            if step >= BATCH_SIZE:
                sample = random.sample(agent.memory, BATCH_SIZE)
                agent.update(sample, step)
                agent.train()


	        # Move to the next state
            state = next_state
        if i < 1000:
        	eps -= 0.001

        	    


        rewards.append(total_reward)
        moy += total_reward
        epsh.append(eps)
        mean_r = np.mean(rewards)
        mean_rh.append(mean_r)
        max_r = np.max(rewards)
        print(f'\repisode {i},score = {total_reward}, epsilon = {eps}')
