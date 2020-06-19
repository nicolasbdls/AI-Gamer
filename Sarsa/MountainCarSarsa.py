import gym
###################################################################################
#This algorithm is the implementation of the SARSA method for the game MountainCar#
###################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from decimal import *

env = gym.make("MountainCar-v0")
env.reset()

episodes = 30000
alpha = 0.1
gamma = 0.99
a = 0.1     # to define epsilon function's offset
b = 7/10        # epsilon function crosses x-axis at b*episodes
eps = a
discretisation = 20
meanreward = 0
indxmean = 0     # from which we start computing mean reward
meanscore = 0
std = 0            # standard deviation
SHOW_EVERY = 100

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) // discrete_os_win_size
    return tuple(discrete_state.astype(
        np.int))  # on utilise ce tuple pour chercher dans les 3 Q values laquelle est bonne pour l'action

def max_action(Q, state, actions=[0,1,2]):
    values = np.array([Q[state, a] for a in actions])
    action = np.argmax(values)
    return action

if __name__=='__main__':

    #env = gym.make("MountainCar-v0")
    DISCRETE_OS_SIZE = [discretisation] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
    action_space = [0, 1, 2]
    states = []
    for s1 in range(discretisation):
        for s2 in range(discretisation):
            states.append((s1, s2))
    Q = {}
    for state in states:
        for action in action_space:
            Q[state, action] = 0
    score = 0

    total_reward = []
    epsilon = []
    finalPosition = np.zeros(episodes)
    finalVelocity = np.zeros(episodes)
    win = 0
    i = 0
    successiveWins = 0

    for j in range(episodes):
    #loop imposing specific number of wins to be achieved in a raw
    #while not win:
        if i % SHOW_EVERY == 0:
            render = True
        else:
            render = False
        i += 1
        obs = env.reset()
        done = False
        score = 0
        state = get_discrete_state(obs)
        action = env.action_space.sample()
        steps = 0
        while not done:
            observation_, reward, done, info = env.step(action)
            state_ = get_discrete_state(observation_)
            action_ = max_action(Q, state_) if np.random.random() > eps else env.action_space.sample()
            score += reward
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[state_, action_] - Q[state, action])
            state = state_
            action = action_
            steps += 1
            if observation_[0] >= 0.5 and observation_[1] >= 0:  # Setting maximum steps to one thousand
                #done = True
                #Q[state, action] = 0
                finalPosition[i-1] = observation_[0]
                finalVelocity[i-1] = observation_[1]
            #if steps >= 1000:
                #done = True

        if i % 100 == 0:
            print('episode', i, 'score', score, 'eps', eps)

        if (finalPosition[i-1] < 0.5 or finalVelocity[i-1]<0):
            win = 0
            successiveWins = 0
        else:
            successiveWins += 1
        #if successiveWins > 100:
            #print('Won', successiveWins, 'times in a row at episode', i, ' with position=', observation_[0], ' and velocity=', observation_[1])
            #win = 1
        total_reward.append(score)
        epsilon.append(eps)
        if eps > 0:
            eps = -a*(1/((b*episodes)*(b*episodes)))*((i*i)-((b*episodes)*(b*episodes))) #second order polynomial decresase
            #eps -= a*1/episodes
        else:
            eps = 0
            if(epsilon[i-2] > 0):
                indxmean = i

    samples = total_reward[indxmean:]
    yy = np.mean(samples)
    print(yy)
    meanscore = round(yy, 2)
    std = round(Decimal(np.std(samples)), 2)
    max = round(Decimal(np.max(samples)), 2)
    print('standard deviation =', std)
    print('Mean score from epsilon = 0 :', meanscore)
    print('Maximum score after epsilon = 0 :', max)
    legend = "Mean score = %.2f \nStandard deviation = %.2s \nMaximum score = %.2f" % (meanscore, std, max)
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Final Reward', color=color)
    plt.ylim(-200, -40)
    ax1.plot(total_reward, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    ax1.text(7.5 / 10 * episodes , -60, legend, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

    plt.axvline(x=7 / 10 * episodes, linewidth=1, color='k', linestyle='--')
    plt.title('Evolution of epsilon and final reward', fontsize=12)
    color = 'tab:orange'
    ax2.set_ylabel('Epsilon', color=color)
    plt.ylim(0, 0.12)
    ax2.plot(epsilon, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

env.close()