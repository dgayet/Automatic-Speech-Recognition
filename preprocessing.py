import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as F
from Classes import *


train_audio_transforms = nn.Sequential(
    T.MelSpectrogram(sample_rate=16e3, n_mels=128),
    T.FrequencyMasking(freq_mask_param=15),
    T.TimeMasking(time_mask_param=35)
)
valid_audio_transforms = T.MelSpectrogram()


def data_processing(data, data_type='train'):
    text_transform = TextTransforms()
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
