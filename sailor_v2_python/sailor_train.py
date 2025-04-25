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


def train(Q, p_alpha, p_gamma, p_epsilon, board_shape):
	num_of_steps_max = int(2.5 * (board_shape[0] + board_shape[1]))
	
	# sum_of_rewards = np.zeros([N_EPISODES], dtype=float)
	for episode in range(N_EPISODES):
		s = np.zeros((2,), dtype=int)
		s[0] = np.random.randint(0, board_shape[0])
		
		the_end = False
		nr_pos = 0
		while not the_end:
			nr_pos += 1  # move count
			
			# Action choosing
			a = pick_action_argmax(Q, s, p_epsilon)
			s_prim, r = sf.environment(s, a, reward_map)
			
			# State-action usability modification:
			Q[s[0], s[1], a - 1] += p_alpha * (r + p_gamma * np.max(Q[s_prim[0], s_prim[1], :]) - Q[s[0], s[1], a - 1])
			
			s = s_prim  # going to the next state
			if nr_pos == num_of_steps_max or s[1] >= board_shape[1] - 1:
				the_end = True
	
	# sum_of_rewards[episode] += reward

if __name__ == '__main__':
    
	for i in range(100):
		np.random.seed(i)
		Q_table = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)
		train(Q_table, ALPHA, GAMMA, EPSILON, reward_map.shape)
		# if episode % 500 == 0:
		# 	print('episode = ' + str(episode) + ' reward = ' + str(sum_of_rewards[episode]))
		
		reward = sf.sailor_test(reward_map, Q_table, 1000)
		print(f"i={i}, r={reward}")
	# sf.draw(reward_map, Q, file_name, reward=r)
