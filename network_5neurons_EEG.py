import os
import requests
import mne
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. URLs for EEG files
files = {
    "vhdr": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-05/ses-1/eeg/sub-05_ses-1_task-rest1_eeg.vhdr?versionId=OjzjIDCLe7A3Um14IszUCs9KgU5cyuYr",
    "eeg": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-05/ses-1/eeg/sub-05_ses-1_task-rest1_eeg.eeg?versionId=yORdvwE3oSKiG9SJp27_8QcaCU0hiCWe",
    "vmrk": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-05/ses-1/eeg/sub-05_ses-1_task-rest1_eeg.vmrk?versionId=6ukKnhMCGp78Erue528BsnY6OJpxRm43"
}

# Download function
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename}")

# Download files if not exists
for key, url in files.items():
    filename = url.split('/')[-1].split('?')[0]
    if not os.path.exists(filename):
        download_file(url, filename)

# -------------------------
# 2. Read EEG data using MNE
vhdr_file = "sub-05_ses-1_task-rest1_eeg.vhdr"
raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

# 3. Convert EEG data to NumPy array
data = raw.get_data()
stimuli = data[:3].T  # use first 3 channels/stimuli

# -------------------------
# 4. Single neuron network (5 neurons)
num_neurons = 5
eta = 0.1
weights = np.random.rand(num_neurons)
outputs = {i: [] for i in range(num_neurons)}

# Hebbian Learning
for x in stimuli:
    for step, xi in enumerate(x):
        for n in range(num_neurons):
            y = xi * weights[n]
            outputs[n].append(y)
            delta_w = eta * xi * y
            weights[n] += delta_w

# -------------------------
# 5. Plot individual neuron outputs
plt.figure(figsize=(12,6))
for n in range(num_neurons):
    plt.plot(outputs[n], label=f'Neuron {n+1}')
plt.title('5-Neuron Network Response to First 3 EEG Stimuli')
plt.xlabel('Step')
plt.ylabel('Output y')
plt.legend()
plt.show()
