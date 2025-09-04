import numpy as np
import matplotlib.pyplot as plt

# Initialize network
num_neurons = 5
eta = 0.1

# Initialize weights randomly for each neuron
weights = np.random.rand(num_neurons)

# Multiple stimuli
stimuli = [
    np.array([1, 0.5, -0.5, 1, -1]),   # stimulus 1
    np.array([0.5, -1, 0.5, -0.5, 1]), # stimulus 2
    np.array([-1, 0.5, 1, -0.5, 0])    # stimulus 3
]

# Store outputs for plotting
outputs = {i: [] for i in range(num_neurons)}

# Apply Hebbian Learning
for x in stimuli:
    for step, xi in enumerate(x):
        for n in range(num_neurons):
            y = xi * weights[n]
            outputs[n].append(y)
            delta_w = eta * xi * y
            weights[n] += delta_w

# Plot results
plt.figure(figsize=(10,5))
colors = ['r', 'g', 'b', 'c', 'm']

for n in range(num_neurons):
    plt.plot(outputs[n], label=f'Neuron {n+1}', color=colors[n])

plt.xlabel('Step')
plt.ylabel('Output y')
plt.title('Small Network (5 Neurons) Response to Multiple Stimuli')
plt.legend()
plt.show()
