import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def plot_histogram(alpha_idx, epsilon_idx, gamma_idx):
	loaded_arr = np.load('data/params_samples-11.npy')
	alpha_linspace = np.linspace(0.001, 0.1, 15)
	alpha = round(alpha_linspace[alpha_idx], 3)

	epsilon_linspace = np.linspace(0.2, 0.8, 15)
	epsilon = round(epsilon_linspace[epsilon_idx], 3)

	gamma_linspace = np.linspace(0.75, 0.99, 10)
	gamma = round(gamma_linspace[gamma_idx], 3)

	data = loaded_arr[int(alpha_idx), int(epsilon_idx), int(gamma_idx), :]

	mean = np.mean(data)
	std = np.std(data)
	max_v = max(data)

	fig, ax = plt.subplots()
	ax.set_xlim(-30, 6)

	# Histogram
	ax.hist(data, bins=int(25), range=(-30, 6))

	# Normal Approximation
	xmin, xmax = ax.get_xlim()
	x = np.linspace(xmin, xmax, 100)
	p = norm.pdf(x, mean, std) * ax.get_ylim()[1] * 9
	p_g6 = round(1 - norm.cdf(6, mean, std), 3)
	# Normal distribution
	ax.fill_between(x, p, color='skyblue', alpha=0.4, label='Normal Approximation')

	# Plot info
	ax.set_title(f"Alpha={alpha}, Epsilon={epsilon}, Gamma={gamma}, P(r>6)={p_g6}")
	ax.set_xlabel("Run Rewards")
	ax.set_ylabel("Frequency")

	ax.text(0.95, 0.95, f"Mean: {mean:.2f}\nStd: {std:.2f}\nMax: {max_v:.2f}",
	        horizontalalignment='right',
	        verticalalignment='top',
	        transform=ax.transAxes,
	        fontsize=10,
	        bbox=dict(facecolor='white', alpha=0.7))

	# ax.legend()

	fig_to_return = fig
	plt.close(fig)
	return fig_to_return


# Define the input and output components
with gr.Blocks() as demo:
	with gr.Row():
		with gr.Column():
			slider_alpha = gr.Slider(minimum=0, maximum=14, step=1, label="Alpha")
			slider_epsilon = gr.Slider(minimum=0, maximum=14, step=1, label="Epsilon")
			slider_gamma = gr.Slider(minimum=0, maximum=9, step=1, label="Gamma")
			button = gr.Button("Submit")

		output = gr.Plot(label="Output Plot")

	# When the button is clicked, run the function
	button.click(fn=plot_histogram, inputs=(slider_alpha, slider_epsilon, slider_gamma), outputs=output)

if __name__ == '__main__':
	# Launch the app
	demo.launch()
