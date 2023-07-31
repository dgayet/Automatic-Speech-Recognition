import torch
import torch.nn as nn
import jiwer
from Decoder import GreedyDecoder

def train_step(model, device, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    train_loss, train_acc = 0, 0
    data_len = len(train_loader.dataset)

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

        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{}] ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len, 100.*batch_idx/len(train_loader), loss.item()
            ))
    train_loss = train_loss / len(train_loader)
    return train_loss

def test_step(model, device, test_loader, criterion, epoch):
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

            if I % 100 == 0 or I == data_test_len:
                print('Test Epoch: {} [{}/{}] ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, I * len(spectrograms), data_test_len, 100.*I/len(test_loader, loss.item())
            ))
                
            for j in range(len(decoded_preds)):
                if I == 0:
                    print("Frase test{}:".format(j))
                    print('Target: \n{}'.format(decoded_targets[j]))
                    print('Prediccion: \n{}'.format(decoded_preds[j]))
                if len(decoded_preds[j]) > 0:
                    test_cer.append(cer(decoded_targets[j], decoded_preds[j]))


def cer(pred, ref):
    return(jiwer.wer(ref,pred))

def train(model, device, train_loader, criterion, optimizer, scheduler, test_loader, epochs):
    for epoch in range(1, epochs + 1):
        train_step(model, device, train_loader, criterion, optimizer, scheduler, epoch)
        if (epoch % 10 == 0):
            test_step(model, device, test_loader, criterion, epoch)
