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
    losses821 = json.load(jsonfile)
with open('training_logs/losses_log_BATCHSIZE_8_RNNL_4.json') as jsonfile:
    losses824 = json.load(jsonfile)
with open('training_logs/losses_log_BATCHSIZE_8_RNNL_5.json') as jsonfile:
    losses825 = json.load(jsonfile)
with open('training_logs/losses_log_BATCHSIZE_8_CNNL_3_RNNL_1.json') as jsonfile:
    losses831 = json.load(jsonfile)
with open('training_logs/losses_log_BATCHSIZE_8_CNNL_5_RNNL_1.json') as jsonfile:
    losses851 = json.load(jsonfile)
with open('training_logs/losses_log_BATCHSIZE_8_CNNL_8_RNNL_1.json') as jsonfile:
    losses881 = json.load(jsonfile)

#%%
[fig,ax] = plt.subplots(figsize=(10,10))
ax.plot(np.arange(1,101,10), losses821[0], linewidth=2, label='CNN Layers=2, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), losses824[0], linewidth=2, label='CNN Layers=2, NRNN Layers=4, Batch Size=8')
ax.plot(np.arange(1,101,10), losses825[0], linewidth=2, label='CNN Layers=2, NRNN Layers=5, Batch Size=8')
ax.plot(np.arange(1,101,10), losses831[0], linewidth=2, label='CNN Layers=3, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), losses851[0], linewidth=2, label='CNN Layers=5, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), losses881[0], linewidth=2, label='CNN Layers=8, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), losses8101[0], linewidth=2, label='CNN Layers=10, NRNN Layers=1, Batch Size=8')


ax.grid()
ax.set_title('Test loss')
ax.legend(fontsize=16)

[fig,ax] = plt.subplots(figsize=(10,10))
ax.plot(losses821[1], linewidth=2, label='CNN Layers=2, NRNN Layers=1, Batch Size=8')
ax.plot(losses824[1], linewidth=2, label='CNN Layers=2, NRNN Layers=4, Batch Size=8')
ax.plot(losses825[1], linewidth=2, label='CNN Layers=2, NRNN Layers=5, Batch Size=8')
ax.plot(losses831[1], linewidth=2, label='CNN Layers=3, NRNN Layers=1, Batch Size=8')
ax.plot(losses851[1], linewidth=2, label='CNN Layers=5, NRNN Layers=1, Batch Size=8')
ax.plot(losses881[1], linewidth=2, label='CNN Layers=8, NRNN Layers=1, Batch Size=8')


ax.set_title("Train loss")
ax.grid()
ax.legend(fontsize=16)
# %%
wer_test821 = np.mean(losses821[2], 1)
wer_test824 = np.mean(losses824[2], 1)
wer_test825 = np.mean(losses825[2], 1)
wer_test831 = np.mean(losses831[2], 1)
wer_test851 = np.mean(losses851[2], 1)
wer_test881 = np.mean(losses881[2], 1)


[fig,ax] = plt.subplots(figsize=(10,10))
ax.plot(np.arange(1,101,10), wer_test821, linewidth=2, label='CNN Layers=2, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), wer_test824, linewidth=2, label='CNN Layers=2, NRNN Layers=4, Batch Size=8')
ax.plot(np.arange(1,101,10), wer_test825, linewidth=2, label='CNN Layers=2, NRNN Layers=4, Batch Size=8')
ax.plot(np.arange(1,101,10), wer_test831, linewidth=2, label='CNN Layers=3, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), wer_test851, linewidth=2, label='CNN Layers=5, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), wer_test881, linewidth=2, label='CNN Layers=8, NRNN Layers=1, Batch Size=8')


ax.set_title("Word Error Rate")
ax.grid()
ax.legend(fontsize=16)

# %% time
# (BS, CNN, RNN)
times = {
    'CNN n=2, RNN n=1, BS=8' : 4100.24,
    'CNN n=2, RNN n=4, BS=8' : 7337.12,
    'CNN n=3, RNN n=1, BS=8' : 5212.45,
    'CNN n=5, RNN n=1, BS=8' : 6913.45,
    'CNN n=8, RNN n=1, BS=8' : 9630.55
}
