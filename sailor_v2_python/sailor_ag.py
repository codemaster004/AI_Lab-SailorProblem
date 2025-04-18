"""
    Genetic algoritm for sailor problem. 
    Sailor sails from left to right side of the map bypassing dangerous places (with negative rewards).
    The environment is nondeterministic due to the wind and waves which means that the boat moves to chosen
    places only with constant probabilities. If you want to evaluate sailor's strategy in the reliable way,
    you need to use many episodes for strategy evaluation.
    
    In this task you need to invent crossover and mutation operations, invent the evaluation method, choose   
    the best parameters values of the evolution, maybe add more parameters and operations. Your solutions 
    should work for different maps and different gamma values.
"""
import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf
import multiprocessing as mp
import json


# np.random.seed(0)

# Evolution parameters ..... (find the best values)
number_of_sumulations = 1000  # do not change (time complexity parameter)
N_INDIVIDUALS = 6  # number of individuals in population (each individual conatains sailor strategy)
N_EPISODES = 20  # number of epizodes for strategy evaluation

P_CROSS = 0.5  # crossover probability
P_MUT = 0.0  # mutation probability
IF_ELITISM = True  # the best individual goes to the next population unchanged
SELECTIVE_PRESSURE = 0.8  # if higher -> more copies of the best individuals in new population expense of worst individuals
N_SPLITS = 5

# Task definition parameters ............ (begin from easy one, and try mid one later)
# file_name = 'map_small.txt'
# file_name = 'map_easy.txt'  # THIS ONE FIRST
# file_name = 'map_mid.txt'
file_name = 'map_big.txt'  # THIS ONE
# file_name = 'map_spiral.txt'
GAMMA = 1.0  # discount factor (part of a task). If gamma < 1, rewards with longer time distance are less important
# (it pays an agent to get positive rewards as soon as possible and penalties as long as possible)

# number fo epochs of evolution with time complexity preservation:
N_EPOCHS = number_of_sumulations // (N_INDIVIDUALS * N_EPISODES)
reward_map = sf.load_data(file_name)  # load map of rewards from file
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(2.5 * (num_of_rows + num_of_columns))  # maximum number of steps in an episode
Popul = np.random.randint(1, 5, (N_INDIVIDUALS, num_of_rows, num_of_columns))  # population of strategies

N_THREADS = 8
best_individuals = {_: [float("-inf"), {}] for _ in range(N_THREADS)}

# print('Initial population = ' + str(Popul))


def fitness_linear(rewards: np.array):
	return -rewards.min() + rewards


def reproduction_roulette(Popul, fitnesses, **kwargs):  # new population based on fittness values
	fit_cum = np.cumsum(fitnesses)
	max_cum_value = fit_cum[-1]
	
	rand_values = np.random.random(size=fitnesses.size) * max_cum_value
	selected_indices = np.searchsorted(fit_cum, rand_values)
	
	return Popul[selected_indices]


def reproduction_rank(population, fitnesses, preassure, **kwargs):
	ranked_indices = np.argsort(fitnesses)[::-1]
	ranks = np.empty_like(ranked_indices)
	ranks[ranked_indices] = np.arange(len(fitnesses))
	
	n = len(population)
	
	rank_probs = ((2 - preassure) / n) + (2 * ranks * (preassure - 1)) / (n * (n - 1))
	
	selected_indices = np.random.choice(
		np.arange(n),
		size=n,
		replace=True,
		p=rank_probs
	)
	
	return population[selected_indices]


def crossover_single_point(parent1: np.ndarray, parent2: np.ndarray):
	shape = parent1.shape
	parent1, parent2 = parent1.flatten(), parent2.flatten()
	
	point = np.random.randint(1, len(parent1))
	child1 = np.concatenate([parent1[:point], parent2[point:]])
	child2 = np.concatenate([parent2[:point], parent1[point:]])
	
	child1.resize(shape)
	child2.resize(shape)
	return child1, child2


def crossover_multi_point(parent1: np.ndarray, parent2: np.ndarray, n_points=2):
	shape = parent1.shape
	parent1, parent2 = parent1.flatten(), parent2.flatten()
	length = len(parent1)
	
	points = np.sort(np.random.choice(np.arange(1, length), size=n_points, replace=False))
	points = np.concatenate([[0], points, [length]])
	
	mask = np.zeros(length, dtype=bool)
	for i in range(0, len(points) - 1, 2):
		mask[points[i]:points[i + 1]] = True
	
	child = np.where(mask, parent1, parent2)
	child.resize(shape)
	
	return child


def crossover_uniform(parent1: np.ndarray, parent2: np.ndarray, p=0.5):
	mask = np.random.rand(*parent1.shape) < p
	child = np.where(mask, parent1, parent2)
	return child


def mutation_uniform(population, mutation_rate, low=1, high=5):
	mutated = population.copy()
	mutation_mask = np.random.rand(*mutated.shape) < mutation_rate
	new_values = np.random.randint(low, high, size=mutated.shape)
	mutated[mutation_mask] = new_values[mutation_mask]
	
	return mutated
	

# np.random.seed(46005)


def main(
		population,
		show=False,
		n_individuals=1,
		n_episodes=1,
		n_epochs=1,
		p_cross=1.0,
		p_mut=1.0,
		selective_pressure=1.0,
		n_splits=1,
		reproduction=None):
	maximum_mean_sum_of_rewards = -1000000000
	
	best_strategy = population[0]
	
	for epoch in range(n_epochs):
		# evaluation of individuals:
		mean_sums_of_rewards = np.zeros([n_individuals])
		minimum_mean_sum_of_rewords = 1000000000
		for individual in range(n_individuals):
			mean_sums_of_rewards[individual] = sf.sailor_test(reward_map, population[individual, ...],
			                                                  n_episodes, GAMMA)
			if minimum_mean_sum_of_rewords > mean_sums_of_rewards[individual]:
				minimum_mean_sum_of_rewords = mean_sums_of_rewards[individual]
			if maximum_mean_sum_of_rewards < mean_sums_of_rewards[individual]:
				maximum_mean_sum_of_rewards = mean_sums_of_rewards[individual]
				best_strategy = population[individual, ...]
				if show:
					print('new best individual = ' + str(maximum_mean_sum_of_rewards) + ' in epoch ' + str(epoch))
		
		# Fittness values must be >= 0 and higher as individual better. You can use selection_factor as an exponent of
		# mean_sums_of_rewards to adjust selection pressure - expected number of copies of the best individuals against the
		# existence of the worst.
		if show:
			print('epoch = ' + str(epoch) + ' avg sum of rewards over population = ' + str(np.mean(mean_sums_of_rewards)))
		
		# for individual in range(number_of_individuals):
		# 	fitness[individual] = 1  # for now ............
		
		# fitness = np.ones([number_of_individuals])
		# fitness = fitness_exp(mean_sums_of_rewards)
		fitness = fitness_linear(mean_sums_of_rewards)
		# selection pressure can be used ....
		# fitness = fitness / (fitness.max() / 10)
		# fitness = np.exp(fitness * selective_pressure)
		
		# rank reproduction can be used ...
		
		# Reproduction ...... can be rank or roulette version
		population = reproduction(population, fitness, selective_pressure)
		# population = reproduction_roulette(population, fitness)
		# population = reproduction_rank(population, fitness)
		
		# Crossover
		for i in range(0, n_individuals - 1, 2):
			if np.random.random() < p_cross:
				# population[i, ...] = crossover_multi_point(population[i], population[i + 1], n_points=n_splits)
				population[i, ...] = crossover_uniform(population[i], population[i + 1])
			if np.random.random() < p_cross:
				# population[i + 1, ...] = crossover_multi_point(population[i + 1], population[i], n_points=n_splits)
				population[i + 1, ...] = crossover_uniform(population[i + 1], population[i])
		
		# Mutation .......
		mutation_uniform(population, p_mut)
	
	# Other operations (elitism, niches, parameter changing functions etc.) ...
	if IF_ELITISM:
		population[np.random.randint(0, n_individuals)] = best_strategy
	
	# end of evolution loop
	
	mean_sum_of_rewards = sf.sailor_test(reward_map, best_strategy, 1000, GAMMA)
	if show:
		print('Average sum of rewards for best strategy = ' + str(mean_sum_of_rewards))
		sf.draw(reward_map, best_strategy, mean_sum_of_rewards)
	return mean_sum_of_rewards


# print('Final population = ' + str(Popul))


def worker_param_search(num, best_indi):
	global best_individuals
	
	for i in range(100):
		N_INDIVIDUALS_ = np.random.randint(4, 20)
		N_EPISODES_ = np.random.randint(4, 5)
		N_EPOCHS_ = number_of_sumulations // (N_INDIVIDUALS_ * N_EPISODES_)
		P_CROSS_ = np.round(np.random.random(), 3)
		P_MUT_ = np.round(np.random.random(), 3) / 4
		SELECTIVE_PRESSURE_ = np.round(np.random.uniform(1.0, 2.0), 3)
		N_SPLITS_ = 0
		REPRODUCTION_ = reproduction_rank
		
		Populs = [
			np.random.randint(1, 5, (N_INDIVIDUALS_, num_of_rows, num_of_columns)),
			np.random.randint(1, 5, (N_INDIVIDUALS_, num_of_rows, num_of_columns)),
			np.random.randint(1, 5, (N_INDIVIDUALS_, num_of_rows, num_of_columns)),
			np.random.randint(1, 5, (N_INDIVIDUALS_, num_of_rows, num_of_columns)),
		]
		
		rt = 0
		for p_ in Populs:
			rt += main(
				p_,
				show=False,
				n_individuals=N_INDIVIDUALS_,
				n_episodes=N_EPISODES_,
				n_epochs=N_EPOCHS_,
				p_cross=P_CROSS_,
				p_mut=P_MUT_,
				selective_pressure=SELECTIVE_PRESSURE_,
				n_splits=N_SPLITS_,
				reproduction=REPRODUCTION_
			)
		
		if rt / len(Populs) > best_individuals[num][0]:
			best_individuals[num][0] = rt / len(Populs)
			best_individuals[num][1] = {
				'N_INDIVIDUALS': N_INDIVIDUALS_,
				'N_EPISODES': N_EPISODES_,
				'P_CROSS': P_CROSS_,
				'P_MUT': P_MUT_,
				'SELECTIVE_PRESSURE': SELECTIVE_PRESSURE_,
				'N_SPLITS': N_SPLITS_,
				'REPRODUCTION': str(REPRODUCTION_),
			}
			print(num, i, best_individuals[num])
	print(best_individuals[num])
	with open('best.json', 'r') as f:
		tmp_best = json.load(f)
	with open('best.json', 'w') as f:
		tmp_best[num] = best_individuals[num]
		json.dump(tmp_best, f)


def worker_seed_search(num):
	with open('best.json', 'r') as f:
		best_indi = json.load(f)
	
	best_reward = float("-inf")
	best_seed = 0
	START_ = 0
	
	N_ = 5
	# for i in range(N_):
	i = num
	while True:
		N_INDIVIDUALS_ = best_indi[str(0)][1]['N_INDIVIDUALS']
		N_EPISODES_ = best_indi[str(0)][1]['N_EPISODES']
		N_EPOCHS_ = number_of_sumulations // (N_INDIVIDUALS_ * N_EPISODES_)
		
		P_CROSS_ = best_indi[str(0)][1]['P_CROSS']
		P_MUT_ = best_indi[str(0)][1]['P_MUT']
		SELECTIVE_PRESSURE_ = best_indi[str(0)][1]['SELECTIVE_PRESSURE']
		
		N_SPLITS_ = 0
		REPRODUCTION_ = reproduction_rank
		
		seed = START_ + i
		
		np.random.seed(seed)
		Popul = np.random.randint(1, 5, (N_INDIVIDUALS_, num_of_rows, num_of_columns))
		
		r = main(
			Popul,
			show=False,
			n_individuals=N_INDIVIDUALS_,
			n_episodes=N_EPISODES_,
			n_epochs=N_EPOCHS_,
			p_cross=P_CROSS_,
			p_mut=P_MUT_,
			selective_pressure=SELECTIVE_PRESSURE_,
			n_splits=N_SPLITS_,
			reproduction=REPRODUCTION_
		)
		
		if r > best_reward:
			best_reward = r
			best_seed = seed
			print('R:', best_reward, 'S:', seed, num, i, best_indi[str(0)])
		
		if r > 0.0:
			print(num, i, best_reward, seed, best_indi[str(0)])
			print("!!!!! HERE !!!!!")
		
		i += N_THREADS
	print(seed, best_reward, best_seed, best_indi[str(num)])


"""
-33.905875
{
	'N_INDIVIDUALS': 18,
	'N_EPISODES': 7,
	'P_CROSS': np.float64(0.948),
	'P_MUT': np.float64(0.384),
	'SELECTIVE_PRESSURE': np.float64(0.282),
	'N_SPLITS': 18,
	'REPRODUCTION': '<function reproduction_roulette at 0x109f5a0e0>'
}
"""

"""
'N_INDIVIDUALS': 16,
'N_EPISODES': 18,
'P_CROSS': 0.877,
'P_MUT': 0.706,
'SELECTIVE_PRESSURE': 1.023,
"""

if __name__ == '__main__':
	# small: 9872, 10060, 25460, 46005
	# big: -14.2685 8638, -11.1025 3662
	
	# processes = []
	# for i in range(N_THREADS):
	# 	p = mp.Process(target=worker_param_search, args=(i, best_individuals))
	# 	processes.append(p)
	# 	p.start()
	# for p in processes:
	# 	p.join()
	# print("All processes completed.")
	
	processes = []
	for i in range(N_THREADS):
		p = mp.Process(target=worker_seed_search, args=(i,))
		processes.append(p)
		p.start()
	for p in processes:
		p.join()
	print("All processes completed.")
	
	# N_INDIVIDUALS_ = 10
	# N_EPISODES_ = 4
	# N_EPOCHS_ = number_of_sumulations // (N_INDIVIDUALS_ * N_EPISODES_)
	# P_CROSS_ = np.float64(0.5)
	# P_MUT_ = np.float64(0.02)
	# SELECTIVE_PRESSURE_ = np.float64(1.59)
	# N_SPLITS_ = 12
	# REPRODUCTION_ = reproduction_rank
	# 
	# np.random.seed(557)
	# 
	# Popul = np.random.randint(1, 5, (N_INDIVIDUALS_, num_of_rows, num_of_columns))
	# r = main(
	# 	Popul,
	# 	show=False,
	# 	n_individuals=N_INDIVIDUALS_,
	# 	n_episodes=N_EPISODES_,
	# 	n_epochs=N_EPOCHS_,
	# 	p_cross=P_CROSS_,
	# 	p_mut=P_MUT_,
	# 	selective_pressure=SELECTIVE_PRESSURE_,
	# 	n_splits=N_SPLITS_,
	# 	reproduction=REPRODUCTION_
	# )
