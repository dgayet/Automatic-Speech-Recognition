import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from preprocessing import *
from Classes import *
from Model import SpeechRecognitionModel
from engine import train
from utils import *


def main():
        # GPU computing
    use_cuda = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_cuda = True
    else:
        device = torch.device("cpu")
    torch.manual_seed(7) 

    # Hiperparameters
    learning_rate=1e-4
    batch_size=8
    epochs=100

    hparams={
        "n_cnn_layers": 8,
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

    print('-------------------------------------------')
    print('device: {}'.format(device))
    print('learning_rate: {}'.format(learning_rate))
    print('batch_size: {}'.format(batch_size))
    print('epochs: {}\n'.format(epochs))

    print('hyperparameters:')
    print(hparams)
    print('-------------------------------------------')
    # Loading Dataset
    train_dataset = Latino40Dataset('../dataset/train.json', '../dataset')
    test_dataset = Latino40Dataset('../dataset/valid.json', '../dataset')

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn= lambda x: data_processing(x, 'train'),
                                **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'),
                                **kwargs)

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats']).to(device)

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                            steps_per_epoch=int(len(train_loader)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy="linear")

    # iter_meter = IterMeter()
    losses = train(model, device, train_loader, criterion, optimizer, scheduler, test_loader, hparams['epochs'])

    losses_filename = '../training_logs/losses_log_BATCHSIZE_{}_CNNL_{}_RNNL_{}.json'.format(batch_size, hparams['n_cnn_layers'], hparams['n_rnn_layers'])
    save_to_json(losses, losses_filename)
    torch.save(model.state_dict(), '../model/model_BS_{}_CNNL_{}_NRNN_{}'.format(batch_size, hparams['n_cnn_layers'], hparams['n_rnn_layers']))

main()
