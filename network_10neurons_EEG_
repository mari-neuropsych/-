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

for key, url in files.items():
    filename = url.split('/')[-1].split('?')[0]
    if not os.path.exists(filename):
        download_file(url, filename)

# -------------------------
# 2. Read EEG data
vhdr_file = "sub-05_ses-1_task-rest1_eeg.vhdr"
raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
data = raw.get_data()
stimuli = data[:3].T  # first 3 channels/stimuli

# -------------------------
# 3. 10-neuron network
num_neurons = 10
eta = 0.1
weights = np.random.rand(num_neurons)
outputs = {i: [] for i in range(num_neurons)}

for x in stimuli:
    for step, xi in enumerate(x):
        for n in range(num_neurons):
            y = xi * weights[n]
            outputs[n].append(y)
            weights[n] += eta * xi * y

# -------------------------
# 4. Plot each neuron separately
plt.figure(figsize=(15, 10))
for n in range(num_neurons):
    plt.subplot(num_neurons, 1, n+1)
    plt.plot(outputs[n], color='b')
    plt.title(f'Neuron {n+1}')
    plt.xlabel('Step')
    plt.ylabel('Output y')
plt.tight_layout()
plt.show()

# 5. Plot average network response
avg_output = np.mean([outputs[n] for n in range(num_neurons)], axis=0)
plt.figure(figsize=(10,5))
plt.plot(avg_output, color='r')
plt.title('Average Response of 10 Neurons')
plt.xlabel('Step')
plt.ylabel('Average Output y')
plt.show()
