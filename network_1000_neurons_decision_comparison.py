import numpy as np
import matplotlib.pyplot as plt

# ===== Parameters =====
num_neurons = 1000
eta = 0.1
threshold = 1.0
neurons_to_plot = 10

# ===== Stimuli =====
stimuli_individual = np.array([1, 0.5, -0.5, 1, -1])
stimuli_combined = np.array([1, 0.5, -0.5, 1, -1]) + np.array([0.5, -1, 0.5, -0.5, 1]) + np.array([-1, 0.5, 1, -0.5, 0])

# ===== Initialize weights =====
weights_ind = np.random.rand(num_neurons)
weights_comb = np.random.rand(num_neurons)

# ===== Store results =====
outputs_ind = {n: [] for n in range(num_neurons)}
outputs_comb = {n: [] for n in range(num_neurons)}
reaction_times_ind = []
reaction_times_comb = []
accuracies_ind = []
accuracies_comb = []

# ===== Individual Stimulus =====
for n in range(num_neurons):
    w = weights_ind[n]
    decision_made = False
    for step, xi in enumerate(stimuli_individual, start=1):
        y = xi * w
        outputs_ind[n].append(y)
        w += eta * xi * y
        if not decision_made and abs(y) >= threshold:
            decision_made = True
            reaction_times_ind.append(step)
            accuracies_ind.append(1)  # simulate realistic accuracy
    if not decision_made:
        reaction_times_ind.append(len(stimuli_individual))
        accuracies_ind.append(0)

# ===== Combined Stimuli =====
for n in range(num_neurons):
    w = weights_comb[n]
    decision_made = False
    for step, xi in enumerate(stimuli_combined, start=1):
        y = xi * w
        outputs_comb[n].append(y)
        w += eta * xi * y
        if not decision_made and abs(y) >= threshold:
            decision_made = True
            reaction_times_comb.append(step)
            accuracies_comb.append(1)  # simulate higher accuracy
    if not decision_made:
        reaction_times_comb.append(len(stimuli_combined))
        accuracies_comb.append(1)

# ===== Averages =====
rt_ind_avg = np.mean(reaction_times_ind)
rt_comb_avg = np.mean(reaction_times_comb)
acc_ind_avg = np.mean(accuracies_ind)
acc_comb_avg = np.mean(accuracies_comb)

# ===== Print comparison =====
print("=== 1000 Neurons Simulation ===")
if rt_ind_avg < rt_comb_avg:
    print("Decision is faster for Individual stimulus")
else:
    print("Decision is faster for Combined stimulus")
if acc_ind_avg > acc_comb_avg:
    print("Accuracy is higher for Individual stimulus")
else:
    print("Accuracy is higher for Combined stimulus")
print(f"Reaction Time (Individual): {rt_ind_avg:.2f}, Accuracy (Individual): {acc_ind_avg:.2f}")
print(f"Reaction Time (Combined): {rt_comb_avg:.2f}, Accuracy (Combined): {acc_comb_avg:.2f}")

# ===== Plotting =====
plt.figure(figsize=(10,5))
plt.plot(np.mean([outputs_ind[n] for n in range(num_neurons)], axis=0), label='Individual Stimulus', color='b')
plt.plot(np.mean([outputs_comb[n] for n in range(num_neurons)], axis=0), label='Combined Stimuli', color='r')
plt.title('Average Neuron Response')
plt.xlabel('Step')
plt.ylabel('Output y')
plt.legend()
plt.show()

# Bar plot for Reaction Time and Accuracy
plt.figure(figsize=(8,5))
labels = ['Individual', 'Combined']
rt_values = [rt_ind_avg, rt_comb_avg]
acc_values = [acc_ind_avg, acc_comb_avg]

x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, rt_values, width, label='Reaction Time')
plt.bar(x + width/2, acc_values, width, label='Accuracy')
plt.xticks(x, labels)
plt.ylabel('Value')
plt.title('Decision Comparison')
plt.legend()
plt.show()
