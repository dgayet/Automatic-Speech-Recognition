#%% imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from Classes import *
from preprocessing import data_processing


#%% plotting an example in time domain and spectrogram
a2db = torchaudio.transforms.AmplitudeToDB()
test_dataset = Latino40Dataset('./dataset/valid.json', './dataset')
n = 450
waveform, sample_rate, label = test_dataset[n]
senial = waveform.data.numpy().squeeze()
[fig,ax] = plt.subplots()

ax.plot(np.arange(len(senial))/sample_rate,senial)
ax.set_title(label)

print('Sample rate: {}Hz'.format(sample_rate))

spectrogram, label, input_length, label_length = data_processing([test_dataset[n]], 'test')
[fig,ax] = plt.subplots()
ax.imshow(a2db(spectrogram.squeeze()))

#%% plotting loss curves
with open('training_logs/losses_log_BATCHSIZE_8_RNNL_1.json') as jsonfile:
    losses81 = json.load(jsonfile)
with open('training_logs/losses_log_BATCHSIZE_8_RNNL_4.json') as jsonfile:
    losses84 = json.load(jsonfile)

[fig,ax] = plt.subplots(figsize=(10,10))
ax.plot(np.arange(1,101,10), losses81[0], linewidth=2, label='test loss, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), losses84[0], linewidth=2, label='test loss, NRNN Layers=4, Batch Size=8')
ax.grid()
ax.legend(fontsize=16)

[fig,ax] = plt.subplots(figsize=(10,10))
ax.plot(losses81[1], linewidth=2, label='train loss, NRNN Layers=1, Batch Size=8')
ax.plot(losses84[1], linewidth=2, label='train loss, NRNN Layers=4, Batch Size=8')
ax.grid()
ax.legend(fontsize=16)
# %%
cer_test81 = np.mean(losses81[2], 1)
cer_test84 = np.mean(losses84[2], 1)

[fig,ax] = plt.subplots(figsize=(10,10))
ax.plot(np.arange(1,101,10), cer_test81, linewidth=2, label='test loss, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), cer_test84, linewidth=2, label='test loss, NRNN Layers=4, Batch Size=8')
ax.grid()
ax.legend(fontsize=16)
