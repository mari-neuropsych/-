import numpy as np
import matplotlib.pyplot as plt

# Initialize weight and learning rate
w = 0.5
eta = 0.1
threshold = 1.0  # threshold to decide
expected_outputs = [1, 1, 1]  # hypothetical expected decision for each stimulus

# Multiple stimuli
stimuli = [
    np.array([1, 0.5, -0.5, 1, -1]),  # stimulus 1
    np.array([0.5, -1, 0.5, -0.5, 1]),  # stimulus 2
    np.array([-1, 0.5, 1, -0.5, 0])  # stimulus 3
]

plt.figure(figsize=(10, 4))
colors = ['r', 'g', 'b']

for i, x in enumerate(stimuli):
    y_values = []
    w_temp = w
    decision_made = False
    reaction_time = None
    accuracy = 0

    for step, xi in enumerate(x, start=1):
        y = xi * w_temp
        y_values.append(y)
        delta_w = eta * xi * y
        w_temp += delta_w

        # Check if decision threshold reached
        if not decision_made and abs(y) >= threshold:
            decision_made = True
            reaction_time = step
            # Check accuracy based on hypothetical expected output
            accuracy = 1 if np.sign(y) == expected_outputs[i] else 0

    print(f'Stimulus {i+1} -> Decision: {decision_made}, Reaction Time: {reaction_time}, Accuracy: {accuracy}')
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
decision_made = False
reaction_time = None
accuracy = 0
expected_output_combined = 1  # hypothetical expected output

for step, xi in enumerate(combined, start=1):
    y = xi * w_temp
    y_values.append(y)
    delta_w = eta * xi * y
    w_temp += delta_w

    if not decision_made and abs(y) >= threshold:
        decision_made = True
        reaction_time = step
        accuracy = 1 if np.sign(y) == expected_output_combined else 0

print(f'Combined Stimuli -> Decision: {decision_made}, Reaction Time: {reaction_time}, Accuracy: {accuracy}')

plt.figure(figsize=(10, 4))
plt.plot(y_values, label='Combined Stimuli', color='m')
plt.xlabel('Step')
plt.ylabel('Output y')
plt.title('Single Neuron Response to Combined Stimuli')
plt.legend()
plt.show()
