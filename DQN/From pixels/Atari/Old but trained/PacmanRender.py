############################################################################
#This algorithm shows the render of pacman for a given model previously    #
#trained with PacmanTrain.ipynb                                            #
############################################################################



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
from collections import deque
from gym import spaces
cv2.ocl.setUseOpenCL(False)
import os
from os.path import exists
import time
from gym import spaces


load = True


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = frame[2:170, :160]
        frame = frame.mean(2)
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = np.reshape(frame, [84, 84, 1])
        return frame

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out
    
    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)
    

def wrap_pytorch(env):
    return ImageToPyTorch(env)         



env_id = "MsPacmanNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env) 
env    = wrap_pytorch(env)


load = True


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
        state = x.to(self.device).float() / 255    #normalize the values
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
        self.actions_memory = torch.from_numpy(np.array(actions)).to(device).unsqueeze(1)  # collect actions
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
        file_name = "trainedmodels/" + "pacman" + ".ptm"
        agent_state = { "episode" : episode,
                        "network" : self.network.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "mean_reward": mean_reward};
        torch.save(agent_state, file_name)
        print("Agent's state saved to ", file_name)

    def load(self):
        self.device = 'cpu'
        file_name = "trainedmodels/" + "pacman" + ".ptm"
        checkpoint = torch.load(file_name, map_location=self.device)
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
    debut = 1
    load = True
    if load == True :
      agent.load()
      debut = 5600

    step = -1
    for i in range(debut, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            env.render()
            time.sleep(0.1)
            step += 1
            action = agent.act(env, state, 0)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            state = next_state

env.close()



