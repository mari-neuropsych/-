import numpy as np
import matplotlib.pyplot as plt

# ==== Flexible network settings ====
num_neurons = 1000
neurons_to_plot = 10
eta = 0.1
threshold = 1.0  # decision threshold
expected_output = 1  # hypothetical expected output for accuracy

# Initialize random weights
weights = np.random.rand(num_neurons)

# Define stimuli
stimuli = [
    np.array([1, 0.5, -0.5, 1, -1]),
    np.array([0.5, -1, 0.5, -0.5, 1]),
    np.array([-1, 0.5, 1, -0.5, 0])
]

# Store outputs and decision metrics
outputs = {i: [] for i in range(num_neurons)}
decisions = {i: False for i in range(num_neurons)}
reaction_times = {i: None for i in range(num_neurons)}
accuracies = {i: 0 for i in range(num_neurons)}

# Hebbian Learning with Decision, Reaction Time, Accuracy
for x in stimuli:
    for step, xi in enumerate(x, start=1):
        for n in range(num_neurons):
            y = xi * weights[n]
            outputs[n].append(y)
            weights[n] += eta * xi * y

            if not decisions[n] and abs(y) >= threshold:
                decisions[n] = True
                reaction_times[n] = step
                accuracies[n] = 1 if np.sign(y) == expected_output else 0

# Print results for first 10 neurons
for n in range(10):
    print(f'Neuron {n+1} -> Decision: {decisions[n]}, Reaction Time: {reaction_times[n]}, Accuracy: {accuracies[n]}')

# Plot selected neurons
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
plt.title(f'Average Response of {num_neurons} Neurons')
plt.xlabel('Step')
plt.ylabel('Average Output y')
plt.show()
