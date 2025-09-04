import os
import requests
import mne
import numpy as np
import matplotlib.pyplot as plt

# 1. URLs for the three EEG files
files = {
    "vhdr": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-05/ses-1/eeg/sub-05_ses-1_task-rest1_eeg.vhdr?versionId=OjzjIDCLe7A3Um14IszUCs9KgU5cyuYr",
    "eeg": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-05/ses-1/eeg/sub-05_ses-1_task-rest1_eeg.eeg?versionId=yORdvwE3oSKiG9SJp27_8QcaCU0hiCWe",
    "vmrk": "https://s3.amazonaws.com/openneuro.org/ds005815/sub-05/ses-1/eeg/sub-05_ses-1_task-rest1_eeg.vmrk?versionId=6ukKnhMCGp78Erue528BsnY6OJpxRm43"
}

# 2. Download function
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename}")

# 3. Download all files if not exist
for key, url in files.items():
    filename = url.split('/')[-1].split('?')[0]
    if not os.path.exists(filename):
        download_file(url, filename)
    else:
        print(f"{filename} already exists, skipping download.")

# 4. Read EEG data using MNE
vhdr_file = "sub-05_ses-1_task-rest1_eeg.vhdr"
raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

# 5. Convert EEG data to NumPy array
data = raw.get_data()
stimuli = data.T  # each row = one stimulus

# 6. Single neuron simulation (first 3 stimuli)
w = 0.5
eta = 0.1

plt.figure(figsize=(10, 4))
for i, x in enumerate(stimuli[:3]):  # exactly first 3 stimuli
    y_values = []
    w_temp = w
    for xi in x:
        y = xi * w_temp
        y_values.append(y)
        delta_w = eta * xi * y
        w_temp += delta_w
    plt.plot(y_values, label=f'Stimulus {i+1}')

plt.xlabel('Step')
plt.ylabel('Output y')
plt.title('Single Neuron Response to First 3 EEG Stimuli')
plt.legend()
plt.show()

# 7. Test combined first 3 stimuli
combined = np.sum(stimuli[:3], axis=0)
y_values = []
w_temp = w
for xi in combined:
    y = xi * w_temp
    y_values.append(y)
    delta_w = eta * xi * y
    w_temp += delta_w

plt.figure(figsize=(10, 4))
plt.plot(y_values, label='Combined First 3 EEG Stimuli', color='m')
plt.xlabel('Step')
plt.ylabel('Output y')
plt.title('Single Neuron Response to Combined First 3 EEG Stimuli')
plt.legend()
plt.show()
