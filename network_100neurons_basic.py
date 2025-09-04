import numpy as np
import matplotlib.pyplot as plt

# Initialize network
num_neurons = 100  # number of neurons
eta = 0.1
weights = np.random.rand(num_neurons)

# Multiple stimuli (simulation data)
stimuli = [
    np.array([1, 0.5, -0.5, 1, -1]),
    np.array([0.5, -1, 0.5, -0.5, 1]),
    np.array([-1, 0.5, 1, -0.5, 0])
]

# Store outputs
outputs = {i: [] for i in range(num_neurons)}

# Hebbian Learning
for x in stimuli:
    for step, xi in enumerate(x):
        for n in range(num_neurons):
            y = xi * weights[n]
            outputs[n].append(y)
            delta_w = eta * xi * y
            weights[n] += delta_w

# Plot individual outputs for first 10 neurons
neurons_to_plot = 10  # sample neurons
plt.figure(figsize=(15, 12))
for n in range(neurons_to_plot):
    plt.subplot(neurons_to_plot, 1, n+1)
    plt.plot(outputs[n], color='b')
    plt.title(f'Neuron {n+1}')
    plt.xlabel('Step')
    plt.ylabel('Output y')
plt.tight_layout()
plt.show()

# Plot average network response
avg_output = np.mean([outputs[n] for n in range(num_neurons)], axis=0)
plt.figure(figsize=(10,5))
plt.plot(avg_output, color='r')
plt.title('Average Response of 100 Neurons')
plt.xlabel('Step')
plt.ylabel('Average Output y')
plt.show()
