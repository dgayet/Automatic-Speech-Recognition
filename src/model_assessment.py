#%% imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from Classes import *
from preprocessing import data_processing
import matplotlib as mpl


#%% plotting an example in time domain and spectrogram
COLOR = 'white'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

a2db = torchaudio.transforms.AmplitudeToDB()
test_dataset = Latino40Dataset('../dataset/valid.json', '../dataset')
n = 450
waveform, sample_rate, label = test_dataset[n]
senial = waveform.data.numpy().squeeze()
[fig,ax] = plt.subplots(figsize=(10,8))

ax.plot(np.arange(len(senial))/sample_rate,senial)
yticks = ax.get_yticklabels()
xticks = ax.get_xticklabels()
ax.set_title(label, fontsize=18, weight='bold')
ax.set_yticklabels(yticks, fontsize=16, weight='bold', )
ax.set_xticklabels(xticks, fontsize=16, weight='bold')

plt.savefig('../figs/demo_signal.png', transparent=True)

print('Sample rate: {}Hz'.format(sample_rate))

#%%
spectrogram, label2, input_length, label_length = data_processing([test_dataset[n]], 'test')
[fig,ax] = plt.subplots(figsize=(10,7))
im = ax.imshow(a2db(spectrogram.squeeze()))
ax.set_title(label, fontsize=16, weight='bold')

yticks = ax.get_yticklabels()
xticks = ax.get_xticklabels()

ax.set_yticklabels(yticks, fontsize=14, weight='bold', )
ax.set_xticklabels(xticks, fontsize=14, weight='bold')
fig.subplots_adjust(right=0.85)

cbar_ax = fig.add_axes([0.88, 0.34, 0.04, 0.31])
fig.colorbar(im, cax=cbar_ax)
plt.savefig('../figs/demo_signal_spectrogram.png', transparent=True, bbox_inches='tight', pad_inches=0)



#%% plotting loss curves
with open('../training_logs/losses_log_BATCHSIZE_8_RNNL_1.json') as jsonfile:
    losses821 = json.load(jsonfile)
with open('../training_logs/losses_log_BATCHSIZE_8_RNNL_4.json') as jsonfile:
    losses824 = json.load(jsonfile)
with open('../training_logs/losses_log_BATCHSIZE_8_RNNL_5.json') as jsonfile:
    losses825 = json.load(jsonfile)
with open('../training_logs/losses_log_BATCHSIZE_8_CNNL_3_RNNL_1.json') as jsonfile:
    losses831 = json.load(jsonfile)
with open('../training_logs/losses_log_BATCHSIZE_8_CNNL_5_RNNL_1.json') as jsonfile:
    losses851 = json.load(jsonfile)
with open('../training_logs/losses_log_BATCHSIZE_8_CNNL_8_RNNL_1.json') as jsonfile:
    losses881 = json.load(jsonfile)
with open('../training_logs/losses_log_BATCHSIZE_8_CNNL_10_RNNL_3.json') as jsonfile:
    losses8103 = json.load(jsonfile)

#%%
COLOR = 'black'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR



[fig,ax] = plt.subplots(figsize=(10,10))
ax.plot(np.arange(1,101,10), losses821[0], linewidth=2, label='CNN Layers=2, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), losses824[0], linewidth=2, label='CNN Layers=2, NRNN Layers=4, Batch Size=8')
ax.plot(np.arange(1,101,10), losses825[0], linewidth=2, label='CNN Layers=2, NRNN Layers=5, Batch Size=8')
ax.plot(np.arange(1,101,10), losses831[0], linewidth=2, label='CNN Layers=3, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), losses851[0], linewidth=2, label='CNN Layers=5, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), losses881[0], linewidth=2, label='CNN Layers=8, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), losses8103[0], linewidth=2, label='CNN Layers=10, NRNN Layers=3, Batch Size=8')
ax.grid()
ax.set_title('Test loss', fontsize=18)
ax.legend(fontsize=16)

plt.savefig('../figs/test_loss.png', bbox_inches='tight', pad_inches=0)


[fig,ax] = plt.subplots(figsize=(10,10))
ax.plot(losses821[1], linewidth=2, label='CNN Layers=2, NRNN Layers=1, Batch Size=8')
ax.plot(losses824[1], linewidth=2, label='CNN Layers=2, NRNN Layers=4, Batch Size=8')
ax.plot(losses825[1], linewidth=2, label='CNN Layers=2, NRNN Layers=5, Batch Size=8')
ax.plot(losses831[1], linewidth=2, label='CNN Layers=3, NRNN Layers=1, Batch Size=8')
ax.plot(losses851[1], linewidth=2, label='CNN Layers=5, NRNN Layers=1, Batch Size=8')
ax.plot(losses881[1], linewidth=2, label='CNN Layers=8, NRNN Layers=1, Batch Size=8')
ax.plot(losses8103[1], linewidth=2, label='CNN Layers=10, NRNN Layers=3, Batch Size=8')
ax.set_title("Train loss", fontsize=18)
ax.grid()
ax.legend(fontsize=16)

plt.savefig('../figs/train_loss.png', bbox_inches='tight', pad_inches=0)

# %%
wer_test821 = np.mean(losses821[2], 1)
wer_test824 = np.mean(losses824[2], 1)
wer_test825 = np.mean(losses825[2], 1)
wer_test831 = np.mean(losses831[2], 1)
wer_test851 = np.mean(losses851[2], 1)
wer_test881 = np.mean(losses881[2], 1)
wer_test8103 = np.mean(losses8103[2], 1)

[fig,ax] = plt.subplots(figsize=(10,10))
ax.plot(np.arange(1,101,10), wer_test821, linewidth=2, label='CNN Layers=2, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), wer_test824, linewidth=2, label='CNN Layers=2, NRNN Layers=4, Batch Size=8')
ax.plot(np.arange(1,101,10), wer_test825, linewidth=2, label='CNN Layers=2, NRNN Layers=5, Batch Size=8')
ax.plot(np.arange(1,101,10), wer_test831, linewidth=2, label='CNN Layers=3, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), wer_test851, linewidth=2, label='CNN Layers=5, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), wer_test881, linewidth=2, label='CNN Layers=8, NRNN Layers=1, Batch Size=8')
ax.plot(np.arange(1,101,10), wer_test8103, linewidth=2, label='CNN Layers=10, NRNN Layers=3, Batch Size=8')


ax.set_title("Character Error Rate", fontsize=18)
ax.grid()
ax.legend(fontsize=16)

plt.savefig('../figs/test_cer.png', bbox_inches='tight', pad_inches=0)

# %% time
# (BS, CNN, RNN)
times = {
    'CNN n=2, RNN n=1, BS=8' : 4100.24/60,
    'CNN n=2, RNN n=4, BS=8' : 7337.12/60,
    'CNN n=3, RNN n=1, BS=8' : 5212.45/60,
    'CNN n=5, RNN n=1, BS=8' : 6913.45/60,
    'CNN n=8, RNN n=1, BS=8' : 9630.55/60
}
times = dict(sorted(times.items(), key=lambda item: item[1]))

parameters = {
    'CNN n=2, RNN n=1, BS=8' : 4779485,
    'CNN n=2, RNN n=4, BS=8' : 18959837,
    'CNN n=2, RNN n=5, BS=8' : 23686621,
    'CNN n=3, RNN n=1, BS=8' : 4798237,
    'CNN n=3, RNN n=4, BS=8' : 18978589,
    'CNN n=5, RNN n=1, BS=8' : 4835741,
    'CNN n=8, RNN n=1, BS=8' : 4891997,
    'CNN n=10, RNN n=3, BS=8' : 14383069
}
parameters = dict(sorted(list(parameters.items()), key=lambda item: item[1]))

[fig,ax] = plt.subplots(figsize=(10,10))
ax.bar(range(len(times)), times.values(), tick_label=times.keys())
ax.set_xticklabels(times.keys(), rotation = 20)
ax.set_title("Tiempo necesario para entrenar (en minutos)", fontsize=15)
plt.savefig('../figs/time.png', bbox_inches='tight', pad_inches=0)


[fig,ax] = plt.subplots(figsize=(10,5))
ax.bar(range(len(parameters)), parameters.values(), tick_label=parameters.keys())
ax.set_xticklabels(parameters.keys(), rotation = 45)
ax.set_title("Parámetros totales de la red (en minutos)", fontsize=15)
plt.savefig('../figs/parameters.png', bbox_inches='tight', pad_inches=0)

#%%
wer_avg ={
    'CNN n=2, RNN n=1, BS=8' : wer_test821[-1],
    'CNN n=2, RNN n=4, BS=8' : wer_test824[-1],
    'CNN n=2, RNN n=5, BS=8' : wer_test825[-1],
    'CNN n=3, RNN n=1, BS=8' : wer_test831[-1],
    'CNN n=5, RNN n=1, BS=8' : wer_test851[-1],
    'CNN n=8, RNN n=1, BS=8' : wer_test881[-1],
    'CNN n=10, RNN n=3, BS=8' : wer_test8103[-1]}
wer_avg = dict(sorted(list(wer_avg.items()), key=lambda item: item[1]))


[fig,ax2] = plt.subplots(figsize=(10,3))
ax2.bar(range(len(wer_avg)), wer_avg.values(), tick_label=wer_avg.keys())
ax2.set_xticklabels(wer_avg.keys(), rotation = 45)
ax2.set_title("CER Promedio", fontsize=15)
plt.savefig('../figs/average_cer.png', bbox_inches='tight', pad_inches=0)
# %%
