########################################################################################
#In this algorithm, the policy is improved by fitting a target function by regression  #
#using the least-squares method. The approximated function is a second order polynomial#
########################################################################################

import gym
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def regression(action,states,T):
    X1 = (np.array(states[action]).T)[0]  #sample positions
    X2 = (np.array(states[action]).T)[1]  #sample velocities
    A = np.array([X1*0+1, X1, X2, X1*X2, X1**2, X2**2]).T  #constructs a second order polynomiql
    coeffs, r, rank, s = np.linalg.lstsq(A, T[action], rcond=None) #implements the least squares method between A and the target a each point
    return coeffs

def getpoly(X, d, e, f):
    Q0= d[0] + d[1]*X[0] + d[2]*X[1] + d[3]*X[0]*X[1] + d[4]*X[0]**2 + d[5]*X[1]**2 
    Q1= e[0] + e[1]*X[0] + e[2]*X[1] + e[3]*X[0]*X[1] + e[4]*X[0]**2 + e[5]*X[1]**2
    Q2= f[0] + f[1]*X[0] + f[2]*X[1] + f[3]*X[0]*X[1] + f[4]*X[0]**2 + f[5]*X[1]**2
    Qarray = Q0, Q1, Q2
    return Qarray


def main():
    env = gym.make('MountainCar-v0')
    env.seed(0)

    action, reward, nxt = [], [], []
    states = [[], [], []]
    T = [[], [], []]
    state = env.reset()

    d = np.ones((6,), dtype=int)
    e = np.ones((6,), dtype=int)
    f = np.ones((6,), dtype=int)

    Qarray = getpoly(np.array(state), d, e, f)
    learning_rate = 0.5
    EPISODES = 3500
    discount_factor = 0.95
    epsilon = 0.5 #hasard entre 0 et 1, a quel point il explore
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // 2 #get an integer
    epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
    compteur = 0
    score = 0

    total_reward = np.zeros(EPISODES)
    epsilontab = []

    for episode in range(EPISODES):
        state = env.reset()
        score = 0
        done = False 
        steps = 0
        while not done:
            compteur+=1
            if np.random.random() > epsilon:
            	action = np.argmax(Qarray) # action depuis Q table
            else:
            	action = np.random.randint(0, env.action_space.n) #random
            states[action].append(state)
            state2, reward, done, _ = env.step(action)
            done = False
            Qarray = getpoly(np.array(state2), d, e, f)
            Qtarget = reward + discount_factor * np.max(Qarray) #target function
            # Add the following line to implement the alternative target function
            #Qtarget = (1 - learning_rate) * Qarray[action] + learning_rate * Qtarget 
            T[action].append(Qtarget)
            state = state2
            score += reward
            steps += 1

            if compteur %20000 == 0:
                if ((np.array(states[0]).T).size!=0 and (np.array(states[1]).T).size!=0 and (np.array(states[2]).T).size!=0) :
                    d = regression(0, states, T)
                    e = regression(1, states, T)
                    f = regression(2, states, T)
                    states = [[], [], []]
                    T = [[], [], []]

            if (state[0] >= env.goal_position):
                done=True
                np.array(Qarray)[action]=0
                print(f'\r won at episode {episode}')

            if (steps >= 1000): #done after 1000 steps
                done =True
                break
               
        epsilontab.append(epsilon)
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        total_reward[episode] = score

        if episode % 100 == 0:
            print(f'\repisode {episode}')

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Final Reward', color=color)
    plt.ylim(-1000, 0)
    ax1.plot(total_reward, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()

    plt.title('Evolution of epsilon and final reward', fontsize=12)

    color = 'tab:orange'

    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(epsilontab, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.show()

    x=np.linspace(-1.2,0.6,100)
    y=np.linspace(-0.07,0.07,100)
    X,Y=np.meshgrid(x,y)
    Z=d[0] + d[1]*X + d[2]*Y + d[3]*X*Y + d[4]*X**2 + d[5]*Y**2 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Qfunction approximation - action 0')
    plt.show()

    x=np.linspace(-1.2,0.6,100)
    y=np.linspace(-0.07,0.07,100)
    X,Y=np.meshgrid(x,y)
    Z=e[0] + e[1]*X + e[2]*Y + e[3]*X*Y + e[4]*X**2 + e[5]*Y**2 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Qfunction approximation - action 1')
    plt.show()

    x=np.linspace(-1.2,0.6,100)
    y=np.linspace(-0.07,0.07,100)
    X,Y=np.meshgrid(x,y)
    Z=f[0] + f[1]*X + f[2]*Y + f[3]*X*Y + f[4]*X**2 + f[5]*Y**2 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Qfunction approximation - action 2')
    plt.show()

if __name__ == '__main__':
    main()