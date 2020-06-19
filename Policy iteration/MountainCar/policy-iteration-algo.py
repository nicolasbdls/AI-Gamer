#########################################################################
#This algorithm is the implementation of policy iteration on MountainCar#
#########################################################################

import gym
import math
import numpy as np
import random

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.8
EPISODES = 1000
SHOW_EVERY = 100
force = 0.001
gravity = 0.0025
max_speed = 0.07
min_position = -1.2
max_position = 0.6
goal_position = 0.5
goal_velocity = 0
discretisation = 20

DISCRETE_OS_SIZE = [discretisation] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) // discrete_os_win_size
    return tuple(discrete_state.astype(
        np.int))

def nxtState(state, action):
    position = state[0] * (max_position - min_position) / discretisation - 1.2
    velocity = state[1] * (max_speed + max_speed) / discretisation - 0.07
    velocity += (action - 1) * force + math.cos(3 * position) * (-gravity)
    position += velocity
    position = np.clip(position, min_position, max_position)
    velocity = np.clip(velocity, -max_speed, max_speed)
    if (position == min_position and velocity < 0): velocity = 0
    done = bool(position >= goal_position and velocity >= goal_velocity)
    reward = -1.0
    position, velocity = get_discrete_state((position, velocity))

    return position, velocity, reward, done, {}

def giveReward(state):
    if (state[0] == WIN_STATE[0]) and (state[1] > 0):
        return 0
    else:
        return -0.01


def policy_evaluation(theta, V):
    print("Policy Evaluation")
    compteur_iteration = 0
    new_state = get_discrete_state(env.reset())
    while True:
        delta = 0
        for x in range(0, discretisation):
            for y in range(0, discretisation):
                V_old[(x, y)] = V[(x, y)]  # pour ne pas avoir le même objet V_old=V
        for x in range(0, discretisation):
            for y in range(0, discretisation):
                state = (x, y)
                position, velocity, reward, done, _ = nxtState(state, policy[state])
                nextstate = (position, velocity)
                if state == WIN_STATE:
                    continue
                reward = giveReward(state)
                v = V[state]
                V[state] = reward + DISCOUNT * V[nextstate]
                delta = max(delta, abs(v - V[state]))

        compteur_iteration += 1
        if delta < theta:
            print('Value table updated', compteur_iteration, 'times')
            return V
            break


def policy_improvement(V, StateAccessible):
    print("policy improvement...")
    policy_stable = True

    for x in range(0, discretisation):
        for y in range(0, discretisation):
            state = (x, y)
            position, velocity, reward, done, _ = nxtState(state, policy[state])
            nextstate = (position, velocity)
            a = policy[state]
            v = V[nextstate]
            a_star = a
            v_star = v
            for action in [0, 1, 2]:
                positionn, velocityy, _, _, _ = nxtState(state, action)
                exploreState = (positionn, velocityy)
                StateAccessible[(exploreState[0], exploreState[1])] = 1
                v_prime = giveReward(state) + DISCOUNT * V[exploreState]
                if v_prime > v_star:
                    print("The state (%d,%s) changed its value" % (state[0], state[1]))
                    a_star = action
                    if policy[(x, y)] != a_star:
                        policy_stable = False
                    v_star = v_prime
                policy[(x, y)] = a_star
    print(StateAccessible)
    return policy_stable


if __name__ == "__main__":

    WIN_STATE = get_discrete_state((goal_position, goal_velocity))

    policy = np.empty(shape=(discretisation, discretisation), dtype=int)
    for x in range(0, discretisation):
        for y in range(0, discretisation):
            policy[(x, y)] = np.random.randint(0, env.action_space.n)
    print(policy)

    V = np.zeros(shape=(discretisation, discretisation))
    for x in range(0, discretisation):
        for y in range(0, discretisation):
            V[(x, y)] = -random.uniform(0, 1)

    StateAccessible = np.zeros(shape=(discretisation, discretisation))
    V[WIN_STATE] = 0  # Verifier s'il s'agit bien du win state
    V_old = np.zeros(shape=(discretisation, discretisation))

    iteration = 0
    policy_stable = False

    while not policy_stable:

        V = policy_evaluation(0.001, V)
        print(V)
        policy_stable = policy_improvement(V, StateAccessible)
        iteration += 1

        if (policy_stable == True):
            print(policy)
            print("Optimal policy found after %d iterations" % iteration)

    env.reset()
    episodes = 3000
    i = 0

    for j in range(episodes): # Render with optimal policy
        done = False
        env.reset()
        action = np.random.randint(0, env.action_space.n)
        while not done:
            env.render()
            observation_, reward, done, info = env.step(action)
            position, velocity = get_discrete_state((observation_[0], observation_[1]))
            #print((position, velocity))
            action = policy[(position, velocity)]

