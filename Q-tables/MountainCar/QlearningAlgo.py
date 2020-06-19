############################################################################
#This algorithms is the implementation of Q-tables for the game MountainCar#
############################################################################



import gym
import numpy as np

env=gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE


epsilon = 0.5 #hasard entre 0 et 1, a quel point il explore
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 #get an integer
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)



q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int)) #we use this tuple to look in the 3 Q-values which is good for the action


for episode in range(EPISODES):

	discrete_state = get_discrete_state(env.reset()) 
	done=False
	while not done:
		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state]) #action from Q table
		else:
			action = np.random.randint(0, env.action_space.n) #random

		new_state, reward, done, _ = env.step(action)

		new_discrete_state = get_discrete_state(new_state)
		
		if not done:
			max_future_q = np.max(q_table[new_discrete_state]) #max Q value in step
			current_q = q_table[discrete_state + (action, )] #current Q value

			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			q_table[discrete_state+(action, )] = new_q #updatee Q table
			
		elif new_state[0] >= env.goal_position:
			
			print(f"Gagné à l'épisode {episode}")
			#np.save(f"qtables/{episode}-qtable.npy", q_table)  #uncomment to save q-table
			q_table[discrete_state + (action, )] = 0
			break
			 
		discrete_state = new_discrete_state

	if episode % 10 == 0:
		print(f'\repisode {episode}')

	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value

env.close()
