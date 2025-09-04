import numpy as np
import matplotlib.pyplot as plt

# Initialize network
num_neurons = 5  # number of neurons
eta = 0.1        # learning rate
weights = np.random.rand(num_neurons)  # random initial weights

# Multiple stimuli (simulation data)
stimuli = [
    np.array([1, 0.5, -0.5, 1, -1]),
    np.array([0.5, -1, 0.5, -0.5, 1]),
    np.array([-1, 0.5, 1, -0.5, 0])
]

# Store outputs for plotting
outputs = {i: [] for i in range(num_neurons)}

# Hebbian Learning
for x in stimuli:
    for step, xi in enumerate(x):
        for n in range(num_neurons):
            y = xi * weights[n]
            outputs[n].append(y)
            delta_w = eta * xi * y
            weights[n] += delta_w

# Plot results
plt.figure(figsize=(10,5))
for n in range(num_neurons):
    plt.plot(outputs[n], label=f'Neuron {n+1}')
plt.title('5-Neuron Network Simulation (Basic)')
plt.xlabel('Step')
plt.ylabel('Output y')
plt.legend()
plt.show()
