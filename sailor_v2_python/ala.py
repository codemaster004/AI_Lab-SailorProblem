import sailor_funct as sf
import numpy as np

file_name = r'C:\Users\alkka\OneDrive\Pulpit\college_py\4sem\ai\lab3\map_easy.txt'
reward_map = sf.load_data(file_name)
gamma = 1.0
number_of_individuals = 90
number_of_episodes_for_eval = 10
number_of_sumulations = 1000
number_of_epochs = number_of_sumulations // (number_of_individuals * number_of_episodes_for_eval)


def reproduction(Popul, fitnesses):
	fit_cum = np.copy(fitnesses)
	for i in range(fitnesses.size - 1):
		fit_cum[i + 1] += fit_cum[i]
	
	max_cum_value = fit_cum[-1]
	Popul_new = np.copy(Popul)
	
	for i in range(fitnesses.size):
		rand_value = np.random.random() * max_cum_value
		for j in range(fit_cum.size):
			prev_val = 0
			if j > 0:
				prev_val = fit_cum[j - 1]
			if (rand_value > prev_val) & (rand_value <= fit_cum[j]):
				Popul_new[i, ...] = np.copy(Popul[j, ...])
				break
	return Popul_new


def run_evolution(reward_map, number_of_individuals, number_of_episodes_for_eval,
                  p_cross, p_mut, selection_pressure, gamma, num_epochs, seed=None, verbose=False):
	if seed is not None:
		np.random.seed(seed)
	
	num_of_rows, num_of_columns = reward_map.shape
	Popul = np.random.randint(1, 5, (number_of_individuals, num_of_rows, num_of_columns))
	
	best_reward = -np.inf
	best_strategy = None
	
	for epoch in range(num_epochs):
		mean_rewards = np.zeros(number_of_individuals)
		fitnesses = np.zeros(number_of_individuals)
		
		for i in range(number_of_individuals):
			mean_rewards[i] = sf.sailor_test(reward_map, Popul[i], number_of_episodes_for_eval, gamma)
			if mean_rewards[i] > best_reward:
				best_reward = mean_rewards[i]
				best_strategy = Popul[i]
				if verbose:
					print(f"[epoch {epoch}] New best: reward = {best_reward}")
		
		fitnesses = (mean_rewards - np.min(mean_rewards) + 1e-5) ** selection_pressure
		
		Popul = reproduction(Popul, fitnesses)
	
	return best_reward, best_strategy


best_overall_reward = -np.inf
best_params = None
print("=== SZUKAMY PARAMETRÓW ===")

for param_iter in range(30000):
	p_cross = np.random.uniform(0.1, 0.9)
	p_mut = np.random.uniform(0.001, 0.1)
	selection_pressure = np.random.uniform(0.5, 3.0)
	
	total_reward = 0
	for _ in range(4):  # 4 różne populacje
		reward, _ = run_evolution(reward_map, number_of_individuals, number_of_episodes_for_eval,
		                          p_cross, p_mut, selection_pressure, gamma, number_of_epochs)
		total_reward += reward
	
	avg_reward = total_reward / 4.0
	
	if avg_reward > best_overall_reward:
		best_overall_reward = avg_reward
		best_params = (p_cross, p_mut, selection_pressure)
		print(
			f"[PARAMS] avg_reward={avg_reward:.2f}, p_cross={p_cross:.3f}, p_mut={p_mut:.4f}, sel_pressure={selection_pressure:.3f}")

p_cross, p_mut, selection_pressure = best_params
best_seed_reward = -np.inf
best_seed = None
print("\n=== SZUKAMY SEEDA ===")

for seed in range(40005):
	reward, strategy = run_evolution(reward_map, number_of_individuals, number_of_episodes_for_eval,
	                                 p_cross, p_mut, selection_pressure, gamma, number_of_epochs, seed=seed)
	if reward > best_seed_reward:
		best_seed_reward = reward
		best_seed = seed
		best_strategy = strategy
		print(f"[SEED] New best seed={seed}, reward={reward}")

print("\n=== NAJLEPSZE PARAMETRY ===")
print(f"p_cross={p_cross}, p_mut={p_mut}, selection_pressure={selection_pressure}")
print(f"best_seed={best_seed}, best_reward={best_seed_reward}")

final_reward = sf.sailor_test(reward_map, best_strategy, 1000, gamma)
sf.draw(reward_map, best_strategy, final_reward)
print(f"Średnia nagroda dla najlepszej strategii: {final_reward}")
