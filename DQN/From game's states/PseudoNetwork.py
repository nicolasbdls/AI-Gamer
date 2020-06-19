###############################################################################################
#In this algorithm a Q-function that has been approximated previously is fitted with a network#
#The loss is printed, different hyperparameters can be easily tuned. In this way, a network   #
#able to fit MountainCar's Q-functions can be obtained                                        #
###############################################################################################

import numpy as np
import torch
import random
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):  # nn model 2 states --> 64layer ---> 64layer ----> 1 Q-value
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 120, bias=True)  # 2d state space
        #self.fc2 = nn.Linear(64, 64, bias=False)
        self.fc3 = nn.Linear(120, action_dim, bias=True)  # one output per action
        self.device = torch.device("cpu")

    def forward(self, x):  # forward function
        state = torch.Tensor(x).to(self.device)
        x = F.relu(self.fc1(state))
        #x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

def f1(x, y): # Q-function
  return -17.853867 + 0.2989069*x -0.81406932*y -1.74610622*x*y +0.27098633*x**2 +33.39477236*y**2

dqn = DQN(2,1)

state = [(random.uniform(-1.2, 0.6), random.uniform(-0.07, 0.07)) for i in range(100)]  #creating 100 random states tuples
value = []
for i in range (100):  #evaluating their value through Q-functiom
  value0 = f1(state[i][0], state[i][1])
  value.append(value0)

optimizer = optim.Adam(dqn.parameters(), lr=0.005)

for t in range(500):
  y_hat = dqn(state)
  optimizer.zero_grad()
  #loss = F.mse_loss(y_hat, torch.FloatTensor(value))  # mse loss
  loss = F.smooth_l1_loss(y_hat, torch.FloatTensor(value)) # huber loss
  print(f'\rstep = {t},loss = {loss}')
  loss.backward()  # backward
  optimizer.step()  # step optimizer -----> improves network