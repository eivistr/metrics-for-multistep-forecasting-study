import os
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

UCR_FOLDER = __location__ + '/data/UCRArchive_2018/'
ICMC_FOLDER = __location__ + '/data/ICMC-USP/'
TRAFFIC_FOLDER = __location__ + '/data/mvdata/traffic.txt'


class UCRDataset(torch.utils.data.Dataset):
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


class ICMCDataset(torch.utils.data.Dataset):
    """

    """

    def __init__(self, filepath, input_size, output_size, train, test_size, transform=None):

        values = pd.read_csv(filepath, delimiter="\n", header=None).values
        train_split = values[:-test_size]
        test_split = values[-test_size:]

        if train:
            self.series = torch.tensor(train_split)
        else:
            self.series = torch.tensor(test_split)

        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.series) - self.input_size - self.output_size

    def __getitem__(self, idx):
        train_seq = self.series[idx:idx+self.input_size]
        target_seq = self.series[idx+self.input_size:idx+self.input_size+self.output_size]
        return train_seq, target_seq


class TrafficDataset(torch.utils.data.Dataset):
    """"""

    def __init__(self, filepath, input_size, output_size, train, test_size, usecol=0, transform=None,):

        values = pd.read_csv(filepath, usecols=[usecol], header=None).values
        train_split = values[:-test_size]
        test_split = values[-test_size:]

        if train:
            self.series = torch.tensor(train_split)
        else:
            self.series = torch.tensor(test_split)

        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.series) - self.input_size - self.output_size

    def __getitem__(self, idx):
        train_seq = self.series[idx:idx+self.input_size]
        target_seq = self.series[idx+self.input_size:idx+self.input_size+self.output_size]
        return train_seq, target_seq


def get_dataset(archive, ds_name, in_size, out_size, test_size=None):

    if archive == "UCR":
        train = UCRDataset(UCR_FOLDER + ds_name + '/' + ds_name + '_TRAIN.tsv', in_size, out_size)
        test = UCRDataset(UCR_FOLDER + ds_name + '/' + ds_name + '_TEST.tsv', in_size, out_size)

    elif archive == "ICMC":
        assert test_size is not None, "Must specify test_frac"
        train = ICMCDataset(ICMC_FOLDER + ds_name + '.data', in_size, out_size, train=True, test_size=test_size)
        test = ICMCDataset(ICMC_FOLDER + ds_name + '.data', in_size, out_size, train=False, test_size=test_size)

    elif ds_name == "Traffic":
        assert test_size is not None, "Must specify test_frac"
        train = TrafficDataset(TRAFFIC_FOLDER, in_size, out_size, train=True, test_size=test_size, usecol=0)
        test = TrafficDataset(TRAFFIC_FOLDER, in_size, out_size, train=False, test_size=test_size, usecol=0)

    else:
        raise ValueError("Invalid dataset")

    return train, test


