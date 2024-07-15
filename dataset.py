import os
import numpy as np
import librosa
from torch.utils.data import Dataset
import torch
import random

# https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-RawNet2/data_utils.py
def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class ASVspoof2019_LA(Dataset):
    def __init__(self, split='train',
                 fake_type=''):
        assert split in ['train', 'val', 'test']
        assert fake_type in ['', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10',
                             'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']
        if split in ['train']:
            assert fake_type == ''

        random.seed(23)

        self.root = './ASVspoof2019/DS_10283_3336/LA'
        # self.aud_format = '.flac'
        self.train_set_len = None

        if split == 'train':
            self.meta_file = os.path.join(self.root, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.train.trn.txt')
            self.dataset_dir = os.path.join(self.root, 'ASVspoof2019_LA_train', 'flac')
            self.mode = 'train'
        elif split == 'test':
            self.mode = 'test'
            self.meta_file = os.path.join(self.root, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.eval.trl.txt')
            self.dataset_dir = os.path.join(self.root, 'ASVspoof2019_LA_eval', 'flac')
        else:
            assert split == 'val'
            self.meta_file = os.path.join(self.root, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.dev.trl.txt')
            self.dataset_dir = os.path.join(self.root, 'ASVspoof2019_LA_dev', 'flac')
            self.mode = 'val'

        with open(self.meta_file, 'r') as f:
            l_meta = f.readlines()

        self.list_IDs = []
        self.labels = {}
        for line in l_meta:
            _, key, _, fm, label = line.strip().split(' ')
            if fake_type == '':
                self.list_IDs.append(key)
                self.labels[key] = 1 if label == 'bonafide' else 0  # 1 is real, 0 is fake
            else:
                if label == 'bonafide' or fm == fake_type:
                    self.list_IDs.append(key)
                    self.labels[key] = 1 if label == 'bonafide' else 0  # 1 is real, 0 is fake


        # following https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-RawNet2/data_utils.py
        self.sample_rate = 16000  # Sample rate of your audio files
        self.cut = 64600  # take ~4 sec audio (64600 samples)


    def __len__(self):
        return len(self.list_IDs)
    @staticmethod
    def min_max_normalize(value, min_value, max_value):
        return (value - min_value) / (max_value - min_value + 0.00000001)

    def __getitem__(self, idx):
        key = self.list_IDs[idx]

        if type(self.dataset_dir) == str:
            X, fs = librosa.load(os.path.join(self.dataset_dir, key + '.flac'), sr=self.sample_rate)
        else:
            if idx < self.train_set_len:
                X, fs = librosa.load(os.path.join(self.dataset_dir[0], key + '.flac'), sr=self.sample_rate)
            else:
                X, fs = librosa.load(os.path.join(self.dataset_dir[1], key + '.flac'), sr=self.sample_rate)

        X_pad = pad(X, self.cut)
        x_inp = torch.Tensor(X_pad)
        y = self.labels[key]

        return x_inp, y

