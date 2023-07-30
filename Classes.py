import os
import json
import torchaudio
from torch.utils.data import Dataset
from charmap import char_map_str

class Latino40Dataset(Dataset):
    def __init__(self, annotations_file, data_root):
        with open(annotations_file) as json_file:
            data_dict = json.load(json_file)
        self.annotation = list(data_dict.values())
        self.data_dir = data_root
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        wav_name = self.annotation[idx]['wav'].replace('{data_root}', self.data_dir)
        waveform, sample_rate = torchaudio.load(wav_name)
        label = self.annotation[idx]['words']
        return waveform, sample_rate, label

class TextTransforms:
    """ maps characteres to integers and vice versa """
    char_map_str = char_map_str
    def __init__(self):
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '
    
    def text_to_int(self, text):
        """ use a character map to convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence
    
    def int_to_text(self, labels):
        """ use a char map to convert an int sequence to a text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('', ' ')
