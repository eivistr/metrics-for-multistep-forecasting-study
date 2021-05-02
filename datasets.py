import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms, utils
import pandas as pd

from sklearn.model_selection import train_test_split


def split_dataset(dataset, test_frac=0.2):
    """Splits a pytorch Dataset object into train and test sets without shuffling."""

    assert 0 < test_frac < 1, 'test fraction must be less than 1'
    train_idx, test_idx = train_test_split(range(len(dataset)), train_size=1-test_frac, shuffle=False)
    train = torch.utils.data.Subset(dataset, train_idx)
    test = torch.utils.data.Subset(dataset, test_idx)
    return train, test


class Traffic(Dataset):
    """"""

    def __init__(self, filepath, input_size, output_size, usecol=0):
        self.series = torch.tensor(pd.read_csv(filepath, usecols=[usecol], header=None).values)
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.series) - self.input_size - self.output_size

    def __getitem__(self, idx):
        train_seq = self.series[idx:idx+self.input_size]
        target_seq = self.series[idx+self.input_size:idx+self.input_size+self.output_size]
        return train_seq, target_seq

    
class UCR18(Dataset):
    """"""
    def __init__(self, filepath, input_size, output_size, transform=None):

        df = pd.read_csv(filepath, delimiter="\t", header=None).drop(0, axis=1).transpose()
        self.series = torch.tensor(df.values)
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return self.series.shape[1]

    def __getitem__(self, idx):
        train_seq = self.series[:self.input_size, idx].unsqueeze(1)
        target_seq = self.series[self.input_size:, idx].unsqueeze(1)
        return train_seq, target_seq


class ICMC(Dataset):
    """

    """

    def __init__(self, filepath, input_size, output_size, transform=None):
        self.series = torch.tensor(pd.read_csv(filepath, delimiter="\n", header=None).values)
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.series) - self.input_size - self.output_size

    def __getitem__(self, idx):
        train_seq = self.series[idx:idx+self.input_size]
        target_seq = self.series[idx+self.input_size:idx+self.input_size+self.output_size]
        return train_seq, target_seq


def test_dataset():
    ds = Traffic("./data/mvdata/traffic.txt", 24, 24, usecol=0)
    traffic_dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, drop_last=True)

    for idx, (seq, target) in enumerate(traffic_dl):
        print('BatchIdx {}, data.shape {}, target.shape {}'.format(idx, seq.shape, target.shape))


if __name__ == "__main__":
    test_dataset()


