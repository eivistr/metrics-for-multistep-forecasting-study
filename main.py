from matplotlib import pyplot as plt
import pandas as pd
import pickle

import numpy as np
import random
import torch

from utilities import Evaluation
from datasets import ICMC, UCR18, Traffic, split_dataset
from models.tcn import run_tcn_model
from models.gru import run_gru_model

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiments_ICMC(ds_name, in_size, out_size, test_frac=0.2, plot=True):

    batch_size = 10
    results = []

    # Load data
    folder_path = './data/ICMC-USP/'
    data = ICMC(folder_path+ds_name+'.data', in_size, out_size)
    train, test = split_dataset(data, test_frac)

    train_dl = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

    # Run TCN model
    x_test, y_test, yhat_test = run_tcn_model(train_dl, test_dl, in_size=in_size, out_size=out_size, epochs=15)
    results.append(Evaluation(x_test, y_test, yhat_test, ds_name, "TCN"))
    if plot:
        results[-1].plot_forecasts(n=3, shuffle=True)

    # Run GRU model
    x_test, y_test, yhat_test = run_gru_model(train_dl, test_dl, out_size=out_size, batch_size=batch_size, epochs=15)
    results.append(Evaluation(x_test, y_test, yhat_test, ds_name, "GRU"))
    if plot:
        results[-1].plot_forecasts(n=3, shuffle=True)

    return results


def run_experiments_UCR(ds_name, in_size, out_size, plot=True):

    batch_size = 10
    results = []

    # Load data
    folder_path = './data/UCRArchive_2018/'+ds_name+'/'
    train = UCR18(folder_path+ds_name+'_TRAIN.tsv', in_size, out_size)
    test = UCR18(folder_path+ds_name+'_TEST.tsv', in_size, out_size)

    train_dl = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

    # Run TCN model
    x_test, y_test, yhat_test = run_tcn_model(train_dl, test_dl, in_size=in_size, out_size=out_size, epochs=15)
    results.append(Evaluation(x_test, y_test, yhat_test, ds_name, "TCN"))
    if plot:
        results[-1].plot_forecasts(n=3, shuffle=True)

    # Run GRU model
    x_test, y_test, yhat_test = run_gru_model(train_dl, test_dl, out_size=out_size, batch_size=batch_size, epochs=15)
    results.append(Evaluation(x_test, y_test, yhat_test, ds_name, "GRU"))
    if plot:
        results[-1].plot_forecasts(n=3, shuffle=True)

    return results


def main():
    # results = run_experiments_UCR(ds_name='ECG5000', in_size=84, out_size=56)

    results = run_experiments_ICMC(ds_name='global', in_size=36, out_size=36)


if __name__ == '__main__':
    main()