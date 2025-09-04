import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import mne

# URLs for EEG files (example: Sub-07)
files = {
    "vhdr": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-07/ses-1/eeg/sub-07_ses-1_task-rest1_eeg.vhdr?versionId=onnVurMZJP734Uj0olOg9IIIOldaXrcb",
    "eeg": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-07/ses-1/eeg/sub-07_ses-1_task-rest1_eeg.eeg?versionId=Bn23difsy3XJAIJ7WoC7tQCUHS9LZ.I4",
    "vmrk": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-07/ses-1/eeg/sub-07_ses-1_task-rest1_eeg.vmrk?versionId=2Q.hj9KlXnJEQNgVYAqFDFwRwNU97fZr"
}

# Temporary download function
def download_file(url):
    local_name = url.split('/')[-1].split('?')[0]
    if not os.path.exists(local_name):
        response = requests.get(url)
        response.raise_for_status()
        with open(local_name, 'wb') as f:
            f.write(response.content)
    return local_name

vhdr_file = download_file(files["vhdr"])
eeg_file = download_file(files["eeg"])
vmrk_file = download_file(files["vmrk"])

# Load EEG data
raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose=False)
data = raw.get_data()[:3, :1000].astype(np.float32)  # first 3 channels, first 1000 samples
data = data.T  # shape: (1000 steps, 3 stimuli)

# Network parameters
num_neurons = 1000
eta = 0.1

# Initialize weights
weights = np.random.rand(num_neurons).astype(np.float32)  # shape: (1000,)

# Store outputs in a 2D array: shape (steps, neurons)
outputs = np.zeros((data.shape[0], num_neurons), dtype=np.float32)

# Hebbian learning using vectorized operations
for t in range(data.shape[0]):
    x = data[t, :]             # shape: (3,)
    # Expand x to match neurons
    x_expanded = np.tile(x, (num_neurons, 1))  # shape: (1000, 3)
    # Compute output: sum over stimuli
    y = np.sum(x_expanded * weights[:, np.newaxis], axis=1)  # shape: (1000,)
    outputs[t, :] = y
    weights += eta * y * np.mean(x)  # update weights vectorized

# Plot first 10 neurons
neurons_to_plot = 10
plt.figure(figsize=(15, 12))
for n in range(neurons_to_plot):
    plt.subplot(neurons_to_plot, 1, n+1)
    plt.plot(outputs[:, n], color='b')
    plt.title(f'Neuron {n+1}')
    plt.xlabel('Step')
    plt.ylabel('Output y')
plt.tight_layout()
plt.show()

# Plot average network response
avg_output = np.mean(outputs, axis=1)
plt.figure(figsize=(10, 5))
plt.plot(avg_output, color='r')
plt.title('Average Response of 1000 Neurons')
plt.xlabel('Step')
plt.ylabel('Average Output y')
plt.show()
