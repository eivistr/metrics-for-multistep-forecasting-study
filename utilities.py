from matplotlib import pyplot as plt
import random
import numpy as np
import pandas as pd
from metrics import calc_tdi_tdm, squared_error, absolute_error

random.seed(0)
np.random.seed(0)


def plot_forecasts(x, y, yhat, n=3, shuffle=True):

    if shuffle:
        indices = random.sample(range(0, x.shape[0]), n)
    else:
        indices = range(0, n)

    index = range(0, x.shape[1] + y.shape[1])
    idx = x.shape[1]

    for i in indices:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 2.5))
        # Forecast
        axs[0].plot(index[:idx], x[i], c='blue', label="Input")
        axs[0].plot(index[idx:], y[i], c='red', label="Truth")
        axs[0].plot(index[idx:], yhat[i], c='green', label="Forecast")
        axs[0].legend()
        axs[0].set_title(f"Prediction")
        # Squared errors
        axs[1].plot(index[idx:], squared_error(y[i], yhat[i]))
        axs[1].set_title(f"Squared errors")
        # Absolute errors
        axs[2].plot(index[idx:], absolute_error(y[i], yhat[i]))
        axs[2].set_title(f"Absolute errors")
        plt.tight_layout()
        plt.show()

        tdi, tdm = calc_tdi_tdm(y[i], yhat[i])
        print(f"TDI: {tdi:.2f}, TDM: {tdm:.2f} ")


class Evaluation:
    """"""

    def __init__(self, x, y, yhat, dataset_name, method_name):

        self.dataset_name = dataset_name
        self.method_name = method_name
        self.x = x
        self.y = y
        self.yhat = yhat

        self.n = y.shape[0]
        self.k = y.shape[1]

        self.sq_err = pd.DataFrame(index=range(self.n), columns=range(self.k), dtype=float)
        self.abs_err = pd.DataFrame(index=range(self.n), columns=range(self.k), dtype=float)

        self.stats = pd.DataFrame(index=range(self.n), columns=["TDI", "TDM"], dtype=float)
        self.accuracy_measures = pd.DataFrame(index=["MSE", "MAE", "RMSE"], columns=range(self.k), dtype=float)

        # Compute metrics for each forecast
        for i in range(self.n):
            tdi, tdm = calc_tdi_tdm(y[i], yhat[i])
            self.stats.loc[i, "TDI"] = tdi
            self.stats.loc[i, "TDM"] = tdm

            self.sq_err.loc[i] = squared_error(y[i], yhat[i])
            self.abs_err.loc[i] = absolute_error(y[i], yhat[i])

        # Calculate mean errors
        self.accuracy_measures.loc["MSE"] = self.sq_err.mean(axis=0)
        self.accuracy_measures.loc["RMSE"] = np.sqrt(self.accuracy_measures.loc["MSE"])

        self.accuracy_measures.loc["MAE"] = self.abs_err.mean(axis=0)

    def plot_forecasts(self, n=3, shuffle=True):

        if shuffle:
            indices = random.sample(range(0, self.x.shape[0]), n)
        else:
            indices = range(0, n)

        index = range(0, self.x.shape[1] + self.y.shape[1])
        idx = self.x.shape[1]

        for i in indices:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 2.5))
            # Forecast
            axs[0].plot(index[:idx], self.x[i], c='blue', label="Input")
            axs[0].plot(index[idx:], self.y[i], c='red', label="Truth")
            axs[0].plot(index[idx:], self.yhat[i], c='green', label="Forecast")
            axs[0].legend()

            tdi, tdm = calc_tdi_tdm(self.y[i], self.yhat[i])
            axs[0].set_title(f"Prediction | TDI: {tdi:.2f} | TDM: {tdm:.2f}")
            # Squared errors
            axs[1].plot(index[idx:], squared_error(self.y[i], self.yhat[i]))
            axs[1].set_title(f"Squared errors")
            # Absolute errors
            axs[2].plot(index[idx:], absolute_error(self.y[i], self.yhat[i]))
            axs[2].set_title(f"Absolute errors")
            plt.tight_layout()
            plt.show()

    def plot_metrics(self):
        # TDI / TDM
        fig = plt.figure()
        fig.add_subplot(121)
        self.stats["TDI"].plot.hist(title="TDI", bins=50, figsize=(15, 3))
        fig.add_subplot(122)
        self.stats["TDM"].plot.hist(title="TDM", bins=50, figsize=(15, 3))
        plt.tight_layout()
        plt.show()

        # Box plots of error measures
        self.sq_err.plot.box(title="Squared errors", figsize=(20, 3))
        plt.show()
        self.abs_err.plot.box(title="Absolute errors", figsize=(20, 3))
        plt.show()

        # Plot mean errors
        self.accuracy_measures.transpose().plot(style='.-', figsize=(20, 3), xticks=range(0, self.k))

    def __str__(self):
        return f"Evaluation of method '{self.method_name}' on dataset '{self.dataset_name}', " \
               f"a total of {self.n} {self.k}-step forecasts.\n"