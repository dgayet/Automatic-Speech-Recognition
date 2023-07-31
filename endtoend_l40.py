#%%
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from preprocessing import *
from Classes import *
from Model import SpeechRecognitionModel
from Decoder import GreedyDecoder

# GPU computing
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
torch.manual_seed(7) 

# Hiperparameters
learning_rate=1e-4
batch_size=5
epochs=100

hparams={
    "n_cnn_layers": 2,
    "n_rnn_layers": 1,
    "rnn_dim": 512,
    "n_class": 29,
    "n_feats": 128,
    "stride": 2,
    "dropout": 0.1,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs
}
#%%
train_dataset = Latino40Dataset('./dataset/train.json', './dataset')
test_dataset = Latino40Dataset('./dataset/valid.json', './dataset')

train_audio_transforms = nn.Sequential(
    T.MelSpectrogram(sample_rate=16e3, n_mels=128),
    T.FrequencyMasking(freq_mask_param=15),
    T.TimeMasking(time_mask_param=35)
)
valid_audio_transforms = T.MelSpectrogram()

text_transform = TextTransforms()

model = SpeechRecognitionModel()
optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
criterion = nn.CTCLoss(blank=28).to(device)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                          steps_per_epoch=int(len(train_loader)),
                                          epochs=hparams['epochs'],
                                          anneal_strategy="linear")

# iter_meter = IterMeter()
for epoch in range(1, epochs +1):
    train(model, device, train_loader, criterion, optimizer, scheduler, epoch)
    if (epoch % 10 == 0):
        test(model, device, test_loader, criterion, epoch)
