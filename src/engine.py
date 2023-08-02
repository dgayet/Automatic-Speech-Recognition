import torch
import torch.nn as nn
import jiwer
import time as t
import numpy as np
from Decoder import GreedyDecoder

def train_step(model, device, train_loader, criterion, optimizer, scheduler, epoch):
    timings = []
    model.train()
    train_loss, train_acc = 0, 0
    data_len = len(train_loader.dataset)

    t0 = t.time()
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms) # (batch, time, n_class)
        output = torch.nn.functional.log_softmax(output, dim=2)
        output = output.transpose(0,1) # ( time, batch, n_class)
        loss = criterion(output, labels, input_lengths, label_lengths)
        train_loss += loss.item() 
        loss.backward()

        optimizer.step()
        scheduler.step()

        dt = t.time() - t0
        if batch_idx % 100 == 0 or batch_idx == len(train_loader):
            print('Train Epoch: {} [{}/{}] ({:.0f}%)]\tloss: {:.6f}\tTime (s): {:.4f}'.format(
                epoch, batch_idx * len(spectrograms), data_len, 100.*batch_idx/len(train_loader), loss.item(), dt
            ))
    return train_loss/len(train_loader)

def test_step(model, device, test_loader, criterion, epoch, step='valid'):
    print('\nevaluating')
    model.eval()
    data_test_len = len(test_loader.dataset)
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for I, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms) # (batch, time, n_class)
            output = nn.functional.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
 
            if (I % 100 == 0 or I == len(test_loader)) and step!='test':
                print('Test Epoch: {} [{}/{}] ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, I * len(spectrograms), data_test_len, 100.*I/len(test_loader), loss.item()
            ))
                
            for j in range(len(decoded_preds)):
                if I == 0 or step=='test':
                    print("Frase test{}:".format(j))
                    print('Target: \n{}'.format(decoded_targets[j]))
                    print('Prediccion: \n{}'.format(decoded_preds[j]))
                if len(decoded_preds[j]) > 0:
                    test_cer.append(cer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer)/len(test_cer)

    print('Test set: Average loss: {:.4f}, Average CER: {:4f}\n'.format(test_loss, avg_cer))
    return test_loss, test_cer

def cer(pred, ref):
    return(jiwer.wer(ref,pred))


def train(model, device, train_loader, criterion, optimizer, scheduler, test_loader, epochs):
    print('training started...')
    t0 = t.time()
    test_loss = []
    train_loss = []
    test_cer = []
    for epoch in range(1, epochs + 1):
        trloss = train_step(model, device, train_loader, criterion, optimizer, scheduler, epoch)
        train_loss.append(trloss)
        if (epoch % 10 == 0):
            tsloss, tscer = test_step(model, device, test_loader, criterion, epoch, 'valid')
            test_loss.append(tsloss)
            test_cer.append(tscer)
    print('Finished training: took {:.2f}s'.format(t.time()-t0))
    return test_loss, train_loss, test_cer

def test(model, device, test_loader, criterion):
    loss, wer = test_step(model, device, test_loader, criterion, 1, 'test')
    return loss, wer
