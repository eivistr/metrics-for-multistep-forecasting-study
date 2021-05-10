from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import pickle
import numpy as np
import random
import torch
import yaml

from evaluation import Result, Evaluation
from datasets import get_dataset
from models.tcn import run_tcn_model
from models.gru import run_gru_model
from models.lstm import run_lstm_model
from models.sarima import run_arima_model
from utilities import validate_cfg

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_models_on_dataset(archive, ds_name, in_size, out_size, batch_size, tcn_epochs, gru_epochs, lstm_epochs,
                          arima_window, arima_m, test_size=None, plot=True):

    results = []

    train, test = get_dataset(archive, ds_name, in_size, out_size, test_size)
    train_dl = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

    # Run TCN model
    x_test, y_test, yhat_test = run_tcn_model(train_dl, test_dl, in_size=in_size, out_size=out_size, epochs=tcn_epochs)
    results.append(Result(x_test, y_test, yhat_test, ds_name, "TCN"))
    if plot:
        results[-1].plot_forecasts()

    # Run GRU model
    x_test, y_test, yhat_test = run_gru_model(train_dl, test_dl, out_size=out_size, epochs=gru_epochs)
    results.append(Result(x_test, y_test, yhat_test, ds_name, "GRU"))
    if plot:
        results[-1].plot_forecasts()

    # Run LSTM model
    x_test, y_test, yhat_test = run_lstm_model(train_dl, test_dl, out_size=out_size, epochs=lstm_epochs)
    results.append(Result(x_test, y_test, yhat_test, ds_name, "GRU"))
    if plot:
        results[-1].plot_forecasts()

    # Run SARIMA model
    train_raw = train.series.squeeze().numpy()
    test_raw = test.series.squeeze().numpy()

    drop_last = len(test) % batch_size  # Shorten test to a multiple of batch_size (eqv. to drop_last = True)
    if drop_last != 0:
        test_raw = test.series.squeeze().numpy()[:-drop_last]

    x_test, y_test, yhat_test = run_arima_model(train_raw,
                                                test_raw,  # drop last
                                                in_size, out_size, arima_window, arima_m)
    results.append(Result(x_test, y_test, yhat_test, ds_name, "ARIMA"))
    if plot:
        results[-1].plot_forecasts()

    return results


def run_experiments(xlsx_file, sheet_name, save_to_dir):
    df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
    # TODO
    # run_experiments(xlsx_file='experiments.xlsx', sheet_name='test', save_to_dir='./results')
    pass


def get_parser():
    """Get parser object."""

    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-f",
        "--file",
        dest="filename",
        help="experiment definition file residing in ./results/",
        metavar="FILE",
        required=True,
    )
    return parser


def main(cfg_file):

    with open(r'./config_experiments/' + cfg_file) as file:
        cfg = yaml.safe_load(file)

    validate_cfg(cfg)

    results = run_models_on_dataset(archive=cfg['archive'],
                                    ds_name=cfg['dataset_name'],
                                    in_size=cfg['in_size'],
                                    out_size=cfg['out_size'],
                                    batch_size=cfg['batch_size'],
                                    tcn_epochs=cfg['tcn_epochs'],
                                    gru_epochs=cfg['gru_epochs'],
                                    lstm_epochs=cfg['lstm_epochs'],
                                    arima_window=cfg['arima_window'],
                                    arima_m=cfg['arima_m'],
                                    test_size=cfg['test_size'])

    with open('./results/' + cfg['save_results_as'], 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args.filename)
