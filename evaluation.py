from matplotlib import pyplot as plt
import random
import numpy as np
import pandas as pd
from metrics import calc_tdi_tdm, squared_error, absolute_error

random.seed(0)
np.random.seed(0)


class Result:
    def __init__(self, x, y, yhat, dataset_name, model_name):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.x = x
        self.y = y
        self.yhat = yhat
        self.n = y.shape[0]
        self.k = y.shape[1]

    def plot_forecasts(self, shuffle=True, grid=(3, 3), figsize=(16, 8)):

        n = grid[0] * grid[1]

        if shuffle:
            indices = random.sample(range(0, self.x.shape[0]), n)
        else:
            indices = range(0, n)

        index = range(0, self.x.shape[1] + self.y.shape[1])
        split_idx = self.x.shape[1]

        fig, ax = plt.subplots(nrows=grid[0], ncols=grid[1], figsize=figsize)

        for i, ax in enumerate(ax.flat):
            tdi, tdm = calc_tdi_tdm(self.y[indices[i]], self.yhat[indices[i]])
            ax.plot(index[:split_idx], self.x[indices[i]], c='blue', label='Input')
            ax.plot(index[split_idx:], self.y[indices[i]], c='red', label='Truth')
            ax.plot(index[split_idx:], self.yhat[indices[i]], c='green', label='Forecast')
            ax.set_title(f"idx {indices[i]} | TDI {tdi:.2f} | TDM {tdm:.2f}", size=10)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=3)
        fig.suptitle(f"{self.model_name} on dataset {self.dataset_name}, {self.k}-step forecasts", size=18)
        plt.tight_layout()
        plt.show()

    def __str__(self):
        return f"Results of {self.model_name} on dataset {self.dataset_name} " \
               f"a total of {self.n} {self.k}-step ahead forecasts.\n"


class Evaluation:
    """"""

    def __init__(self, result):

        self.result = result
        self.dataset_name = result.dataset_name
        self.model_name = result.model_name
        self.n = result.n
        self.k = result.k

        self.sq_err = pd.DataFrame(index=range(self.n), columns=range(self.k), dtype=float)
        self.abs_err = pd.DataFrame(index=range(self.n), columns=range(self.k), dtype=float)

        self.stats = pd.DataFrame(index=range(self.n), columns=["TDI", "TDM"], dtype=float)
        self.accuracy_measures = pd.DataFrame(index=["MSE", "MAE", "RMSE"], columns=range(self.k), dtype=float)

        # Compute metrics for each forecast
        for i in range(self.n):
            tdi, tdm = calc_tdi_tdm(self.result.y[i], self.result.yhat[i])
            self.stats.loc[i, "TDI"] = tdi
            self.stats.loc[i, "TDM"] = tdm

            self.sq_err.loc[i] = squared_error(self.result.y[i], self.result.yhat[i])
            self.abs_err.loc[i] = absolute_error(self.result.y[i], self.result.yhat[i])

        # Calculate mean errors
        self.accuracy_measures.loc["MSE"] = self.sq_err.mean(axis=0)
        self.accuracy_measures.loc["RMSE"] = np.sqrt(self.accuracy_measures.loc["MSE"])

        self.accuracy_measures.loc["MAE"] = self.abs_err.mean(axis=0)

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

    def plot_forecasts(self, shuffle=True, grid=(3, 3), figsize=(16, 8)):
        self.result.plot_forecasts(shuffle=shuffle, grid=grid, figsize=figsize)

    def __str__(self):
        return f"Evaluation of {self.model_name} on dataset {self.dataset_name}, " \
               f"a total of {self.n} {self.k}-step ahead forecasts.\n"
