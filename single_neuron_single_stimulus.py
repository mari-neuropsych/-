import numpy as np
import matplotlib.pyplot as plt

# Initialize weight and learning rate
w = 0.5
eta = 0.1

# Multiple stimuli
stimuli = [
    np.array([1, 0.5, -0.5, 1, -1]),  # stimulus 1
    np.array([0.5, -1, 0.5, -0.5, 1]),  # stimulus 2
    np.array([-1, 0.5, 1, -0.5, 0])  # stimulus 3
]

# Test each stimulus individually
plt.figure(figsize=(10, 4))
colors = ['r', 'g', 'b']

for i, x in enumerate(stimuli):
    y_values = []
    w_temp = w  # temporary weight for this stimulus

    for xi in x:
        y = xi * w_temp
        y_values.append(y)
        delta_w = eta * xi * y
        w_temp += delta_w

    plt.plot(y_values, label=f'Stimulus {i + 1}', color=colors[i])

plt.xlabel('Step')
plt.ylabel('Output y')
plt.title('Single Neuron Response to Individual Stimuli')
plt.legend()
plt.show()

# Test all stimuli combined
combined = sum(stimuli)
y_values = []
w_temp = w

for xi in combined:
    y = xi * w_temp
    y_values.append(y)
    delta_w = eta * xi * y
    w_temp += delta_w

plt.figure(figsize=(10, 4))
plt.plot(y_values, label='Combined Stimuli', color='m')
plt.xlabel('Step')
plt.ylabel('Output y')
plt.title('Single Neuron Response to Combined Stimuli')
plt.legend()
plt.show()
