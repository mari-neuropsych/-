import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import mne

# 1. EEG file URLs for one subject
files = {
    "vhdr": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-10/ses-1/eeg/sub-10_ses-1_task-rest1_eeg.vhdr?versionId=7aKMcnaEOFylFvwaJ5PgwDvVQeuTqA9D",
    "eeg": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-10/ses-1/eeg/sub-10_ses-1_task-rest1_eeg.eeg?versionId=gXOT5Z6XTF_vQPbBHZPIvnOt7aVK_PH0",
    "vmrk": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-10/ses-1/eeg/sub-10_ses-1_task-rest1_eeg.vmrk?versionId=KBTkr4ssLjBWBTabYIaIU3KKMXiCDf.Q"

}

# 2. Function to download files temporarily
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

# 3. Load EEG data
raw = mne.io.read_raw_brainvision(local_files["vhdr"], preload=True, verbose=False)
data = raw.get_data()[:3, :1000]  # use first 3 channels and first 1000 samples
data = data.astype(np.float32)

# 4. Initialize network
num_neurons = 1000
eta = 0.1
weights = np.random.rand(num_neurons).astype(np.float32)
outputs = np.zeros((data.shape[1], num_neurons), dtype=np.float32)  # shape (steps, neurons)

# 5. Hebbian learning with vectorization
for t in range(data.shape[1]):
    x = data[:, t][:, np.newaxis]  # shape (3,1)
    y = x.T @ weights[:3]          # multiply 3x1 with first 3 weights for simplicity
    outputs[t, :] = y              # replicate if needed
    weights += eta * y             # update weights

# 6. Plot first 10 neurons
plt.figure(figsize=(15, 12))
for n in range(10):
    plt.subplot(10, 1, n+1)
    plt.plot(outputs[:, n], color='b')
    plt.title(f'Neuron {n+1}')
    plt.xlabel('Step')
    plt.ylabel('Output y')
plt.tight_layout()
plt.show()

# 7. Plot average network response
avg_output = outputs.mean(axis=1)
plt.figure(figsize=(10,5))
plt.plot(avg_output, color='r')
plt.title(f'Average Response of {num_neurons} Neurons')
plt.xlabel('Step')
plt.ylabel('Average Output y')
plt.show()
