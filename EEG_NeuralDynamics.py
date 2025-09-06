# 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt
import requests
import mne

# 2. Download EEG files
files = {
    "vhdr": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-10/ses-1/eeg/sub-10_ses-1_task-rest1_eeg.vhdr?versionId=7aKMcnaEOFylFvwaJ5PgwDvVQeuTqA9D",
    "eeg": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-10/ses-1/eeg/sub-10_ses-1_task-rest1_eeg.eeg?versionId=gXOT5Z6XTF_vQPbBHZPIvnOt7aVK_PH0",
    "vmrk": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-10/ses-1/eeg/sub-10_ses-1_task-rest1_eeg.vmrk?versionId=KBTkr4ssLjBWBTabYIaIU3KKMXiCDf.Q"
}

def download_file(url, local_name):
    response = requests.get(url)
    response.raise_for_status()
    with open(local_name, "wb") as f:
        f.write(response.content)
    return local_name

local_files = {}
for key, url in files.items():
    filename = url.split("/")[-1].split("?")[0]
    local_files[key] = download_file(url, filename)

# 3. Read EEG data
raw = mne.io.read_raw_brainvision(local_files["vhdr"], preload=True, verbose=False)
data = raw.get_data().astype(np.float32)
num_channels, num_steps = data.shape

# 3.1 Normalize data
data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-6)

# 4. Create neural network
num_neurons = 1000
eta = 0.0005
weights = np.random.rand(num_neurons, num_channels).astype(np.float32)
outputs = np.zeros((num_steps, num_neurons), dtype=np.float32)

# 5. Hebbian Learning with nonlinearity and noise
for t in range(num_steps):
    x = data[:, t]
    noise = np.random.normal(0, 0.05, size=num_neurons)
    y = np.tanh(np.dot(weights, x)) + noise
    outputs[t, :] = y
    weights += eta * (y[:, np.newaxis] * x[np.newaxis, :])

# 6. Compute cognitive and decision indicators
threshold = 0.5
strength = np.sum(outputs > threshold, axis=1) / num_neurons
accuracy = 1 - np.std(outputs, axis=1) / np.max(np.std(outputs, axis=1))
reaction_time = np.argmax(strength > threshold)
decision_output = outputs[reaction_time, :]

# 6.1 Print the values
print(f"Neural Strength for each time step:\n{strength}\n")
print(f"Decision Accuracy for each time step:\n{accuracy}\n")
print(f"Reaction Time (step index): {reaction_time}\n")
print(f"Decision Output at reaction time:\n{decision_output}\n")

# 7. Plot first 10 neurons
plt.figure(figsize=(15, 12))
for n in range(10):
    plt.subplot(10, 1, n+1)
    plt.plot(outputs[:, n], color='b')
    plt.title(f'Neuron {n+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Output y')
plt.tight_layout()
plt.show()

# 8. Plot cognitive and decision indicators
plt.figure(figsize=(14,6))
plt.plot(strength, label='Neural Strength', color='r')
plt.plot(accuracy, label='Decision Accuracy', color='g')
plt.title('EEG-Based Cognitive Indicators')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()

# 9. Heatmap of neural activity
plt.figure(figsize=(15,8))
plt.imshow(outputs.T, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Neuron Output')
plt.title('Heatmap of Neural Activity (1000 Neurons)')
plt.xlabel('Time Step')
plt.ylabel('Neuron Index')
plt.show()
