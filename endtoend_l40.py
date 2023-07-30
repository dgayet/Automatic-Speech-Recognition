#%%
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from Classes import *

#%%
train_dataset = Latino40Dataset('./dataset/train.json', './dataset')
test_dataset = Latino40Dataset('./dataset/valid.json', './dataset')

#%%
train_audio_transforms = nn.Sequential(
    T.MelSpectrogram(sample_rate=16e3, n_mels=128),
    T.FrequencyMasking(freq_mask_param=15),
    T.TimeMasking(time_mask_param=35)
)
valid_audio_transforms = T.MelSpectrogram()

text_transform = TextTransforms()

def data_processing(data, data_type='train'):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0,1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0,1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))
    
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2,3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths
#%%
waveform, sample_rate, label = test_dataset[0]
signal = waveform.data.numpy().squeeze()

[fig, ax] = plt.subplots()
ax.plot(np.arange(len(signal))/sample_rate, signal)
ax.set_title(label)

print('Sample Rate: {}Hz'.format(sample_rate))
# %%
