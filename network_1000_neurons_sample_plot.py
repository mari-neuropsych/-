import numpy as np
import matplotlib.pyplot as plt

# ==== Flexible network settings ====
num_neurons = 1000          # Total number of neurons
neurons_to_plot = 10        # Number of neurons to display in subplots
eta = 0.1                   # Learning rate

# Initialize random weights
weights = np.random.rand(num_neurons)

# Define stimuli (can add more or modify)
stimuli = [
    np.array([1, 0.5, -0.5, 1, -1]),
    np.array([0.5, -1, 0.5, -0.5, 1]),
    np.array([-1, 0.5, 1, -0.5, 0])
]

# Store outputs for each neuron
outputs = {i: [] for i in range(num_neurons)}

# Hebbian Learning for each stimulus
for x in stimuli:
    for step, xi in enumerate(x):
        for n in range(num_neurons):
            y = xi * weights[n]
            outputs[n].append(y)
            weights[n] += eta * xi * y  # Update weight

# ==== Plot selected neurons in separate subplots ====
plt.figure(figsize=(15, 12))
for n in range(neurons_to_plot):
    plt.subplot(neurons_to_plot, 1, n+1)
    plt.plot(outputs[n], color='b')
    plt.title(f'Neuron {n+1}')
    plt.xlabel('Step')
    plt.ylabel('Output y')
plt.tight_layout()
plt.show()

# ==== Plot average network response ====
avg_output = np.mean([outputs[n] for n in range(num_neurons)], axis=0)
plt.figure(figsize=(10,5))
plt.plot(avg_output, color='r')
plt.title(f'Average Response of {num_neurons} Neurons')
plt.xlabel('Step')
plt.ylabel('Average Output y')
plt.show()
