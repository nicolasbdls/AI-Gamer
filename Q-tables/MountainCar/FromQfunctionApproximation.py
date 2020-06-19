###########################################################################################################
#In this algorithm, the Q-table of a winning episode is uploaded. The agent's policy is extracted from it.#
#3 Q-functions are approximated from the Q-table and then the agent takes the action which corresponds to #
#the argmax between the 3 Q-functions.                                                                     #
###########################################################################################################


import gym
import numpy as np

env=gym.make("MountainCar-v0")
env.reset()

EPISODES = 1000

SHOW_EVERY = 1000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE


i=900
q_table = np.load(f"qtables/{i}-qtable.npy")
a,b,c = np.dsplit(q_table, 3)
z1=np.amin(a)
z2=np.amin(b)
z3=np.amin(c)
z4=np.amax(a)
z5=np.amax(b)
z6=np.amax(c)
a.shape=(20, 20)
b.shape=(20, 20)
c.shape=(20, 20)

x=np.linspace(0,20,20)
y=np.linspace(0,20,20)



X, Y = np.meshgrid(x, y, copy=False)
X=X.flatten()
Y=Y.flatten()
A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
B=a.flatten()
C=b.flatten()
D=c.flatten()
d, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
e, r, rank, s = np.linalg.lstsq(A, C, rcond=None)
f, r, rank, s = np.linalg.lstsq(A, D, rcond=None)



def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int)) #on utilise ce tuple pour chercher dans les 3 Q values laquelle est bonne pour l'action

def get_poly(state):
	poly1 = d[0] + d[1]*state[0] + d[2]*state[1] + d[3]*state[0]**2 + d[4]*state[0]**2*state[1] + d[5]*state[0]**2*state[1]**2 + d[6]*state[1]**2 + d[7]*state[0]*state[1]**2 + d[8]*state[0]*state[1]
	poly2 = e[0] + e[1]*state[0] + e[2]*state[1] + e[3]*state[0]**2 + e[4]*state[0]**2*state[1] + e[5]*state[0]**2*state[1]**2 + e[6]*state[1]**2 + e[7]*state[0]*state[1]**2 + e[8]*state[0]*state[1]
	poly3 = f[0] + f[1]*state[0] + f[2]*state[1] + f[3]*state[0]**2 + f[4]*state[0]**2*state[1] + f[5]*state[0]**2*state[1]**2 + f[6]*state[1]**2 + f[7]*state[0]*state[1]**2 + f[8]*state[0]*state[1]
	polyarray = [poly1, poly2, poly3]
	return polyarray



for episode in range(EPISODES):



	discrete_state = get_discrete_state(env.reset()) 
	done=False
	while not done:
		env.render()
		po=get_poly(discrete_state)
		action = np.argmax(po) 

	
		new_state, reward, done, _ = env.step(action)

		new_discrete_state = get_discrete_state(new_state)


		if new_state[0] >= env.goal_position:
			print(f"Gagné à l'épisode {episode}")
			break
			 

		discrete_state = new_discrete_state

env.close()