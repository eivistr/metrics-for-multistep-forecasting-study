import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def split_dataset(dataset, test_frac=0.2):
    """"""

    assert test_frac < 1
    # Initial split into train and val
    train_idx, test_idx = train_test_split(np.arange(len(dataset)), train_size=1-test_frac, shuffle=False)
    train = torch.utils.data.Subset(dataset, train_idx)
    test = torch.utils.data.Subset(dataset, test_idx)
    return train, test


class Traffic(Dataset):
    """"""

    def __init__(self, csv_file, window, horizon, usecol=0):
        self.series = torch.tensor(pd.read_csv(csv_file, usecols=[usecol], header=None).values)
        self.window = window
        self.horizon = horizon

    def __len__(self):
        return len(self.series) - self.window - self.horizon

    def __getitem__(self, idx):
        train_seq = self.series[idx:idx+self.window]
        target_seq = self.series[idx+self.window:idx+self.window+self.horizon]
        return train_seq, target_seq

    
class ECG5000(Dataset):
    """"""
    def __init__(self, tsv_file, window=84, horizon=56, transform=None):
        
        df = pd.read_csv(tsv_file, delimiter="\t", header=None)
        df = df.drop(0, axis=1).transpose()
        self.series = torch.tensor(df.values)
        self.window = window
        self.horizon = horizon

    def __len__(self):
        return self.series.shape[1]

    def __getitem__(self, idx):
        train_seq = self.series[:self.window, idx].unsqueeze(1)
        target_seq = self.series[self.window:, idx].unsqueeze(1)
        return train_seq, target_seq


def test():
    ds = Traffic("./data/mvdata/traffic.txt", 24, 24, usecol=0)
    traffic_dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, drop_last=True)

    for idx, (seq, target) in enumerate(traffic_dl):
        print('BatchIdx {}, data.shape {}, target.shape {}'.format(idx, seq.shape, target.shape))


if __name__ == "__main__":
    test()


