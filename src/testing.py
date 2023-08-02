#%%
import torch
import torch.nn as nn
import torchaudio
import sys
import torch.utils.data as data
sys.path.append('./src/')
from Model import SpeechRecognitionModel
from engine import test
from Classes import Latino40Dataset
from preprocessing import data_processing


use_cuda = False
if torch.cuda.is_available():
    device = torch.device("cuda")
    use_cuda = True
else:
    device = torch.device("cpu")
torch.manual_seed(7) 

learning_rate=1e-4
batch_size=8
epochs=100

hparams={
    "n_cnn_layers": 2,
    "n_rnn_layers": 5,
    "rnn_dim": 512,
    "n_class": 29,
    "n_feats": 128,
    "stride": 2,
    "dropout": 0.1,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs
}


model_filename = '../model/model_BS_{}_CNNL_{}_NRNN_{}'.format(
    batch_size, 
    hparams['n_cnn_layers'],
    hparams['n_rnn_layers'])

print('Model name: {}'.format(model_filename))
print(hparams)


model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats']).to(device)

if use_cuda == True:
    model.load_state_dict(torch.load(model_filename))
else:
    checkpoint = torch.load(model_filename, map_location=torch.device('cpu'))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint, strict=False)


test_dataset = Latino40Dataset('../dataset/valid.json', '../dataset')
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
test_loader = data.DataLoader(dataset=test_dataset,
                            batch_size=hparams['batch_size'],
                            shuffle=False,
                            collate_fn=lambda x: data_processing(x, 'valid'),
                            **kwargs)
criterion = nn.CTCLoss(blank=28).to(device)

model.eval()
total_params = sum(p.numel() for p in model.parameters())
avg_loss, wer = test(model, device, test_loader, criterion)

print('---------------------------------------------------------------------------------')
print('\nSTATISTICS:\n')
print('Average loss: {}'.format(avg_loss))
print('Word Error Rate vector:\n[')
[print(w) for w in wer]
print(']\n---------------------------------------------------------------------------------')
