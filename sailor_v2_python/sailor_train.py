import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct_q as sf

N_EPISODES = 4_000  # The number of training epizodes DO NOT INCREASE!
GAMMA = 0.8  # discount factor

# Training data parameters - these may be time-dependent functions:
ALPHA = 0.09

EPSILON = 0.63
TEMP = 1

# file_name = 'map_small.txt'
# file_name = 'map_easy.txt'
file_name = 'map_simple.txt'
# file_name = 'map_middle.txt'
# file_name = 'map_mid.txt'
# file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape


def pick_action_argmax(Q, state, epsilon):
	if np.random.rand() < epsilon:
		a = np.random.randint(0, 5)
	else:
		a = np.argmax(Q[state[0], state[1], :])
	return a


def pick_action_softmax(Q, state, temp):
	max_a_v = np.max(Q[state[0], state[1], :])
	soft_a = np.exp((Q[state[0], state[1], :] - max_a_v) / temp)
	soft_a /= soft_a.sum()
	return np.random.choice(np.arange(len(soft_a)), 1, p=soft_a)[0]

def train():
	num_of_steps_max = int(2.5 * (num_of_rows + num_of_columns))  # maximum number of steps in an episode
	Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  # trained usability table of <state,action> pairs
	sum_of_rewards = np.zeros([N_EPISODES], dtype=float)
	
	for episode in range(N_EPISODES):
		state = np.zeros((2,), dtype=int)
		state[0] = np.random.randint(0, num_of_rows)
		
		the_end = False
		nr_pos = 0
		# reward_map_curr = reward_map
		while not the_end:
			nr_pos += 1  # move count
			
			# Action choosing
			# action = pick_action_softmax(Q, state, TEMP)
			action = pick_action_argmax(Q, state, EPSILON)
			
			state_next, reward = sf.environment(state, action, reward_map)
			
			# State-action usability modification:
			Q[state[0], state[1], action - 1] += ALPHA * (
					reward + GAMMA * np.max(Q[state_next[0], state_next[1], :]) - Q[state[0], state[1], action - 1])
			
			state = state_next  # going to the next state
			
			if (nr_pos == num_of_steps_max) | (state[1] >= num_of_columns - 1):
				the_end = True
			
			sum_of_rewards[episode] += reward

for i in range(100):
	np.random.seed(i)
	
	
	# if episode % 500 == 0:
	# 	print('episode = ' + str(episode) + ' reward = ' + str(sum_of_rewards[episode]))

	r = sf.sailor_test(reward_map, Q, 1000)
	print(f"i={i}, r={r}")
	# sf.draw(reward_map, Q, file_name, reward=r)
