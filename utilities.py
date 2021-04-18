from matplotlib import pyplot as plt
import random
import numpy as np
import pandas as pd
from metrics import calc_tdi_tdm, squared_error, absolute_error

random.seed(0)
np.random.seed(0)


def plot_forecasts(x, y, yhat, n=5, shuffle=True):

    if shuffle:
        indices = random.sample(range(0, x.shape[0]), n)
    else:
        indices = range(0, n)

    index = range(0, x.shape[1] + y.shape[1])
    l = x.shape[1]

    for i in indices:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 2.5))
        # Forecast
        axs[0].plot(index[:l], x[i], c='blue', label="Input")
        axs[0].plot(index[l:], y[i], c='red', label="Truth")
        axs[0].plot(index[l:], yhat[i], c='green', label="Forecast")
        axs[0].legend()
        axs[0].set_title(f"Prediction")
        # Squared errors
        axs[1].plot(index[l:], squared_error(y[i], yhat[i]))
        axs[1].set_title(f"Squared errors")
        # Absolute errors
        axs[2].plot(index[l:], absolute_error(y[i], yhat[i]))
        axs[2].set_title(f"Absolute errors")
        plt.tight_layout()
        plt.show()

        tdi, tdm = calc_tdi_tdm(y[i], yhat[i])
        print(f"TDI: {tdi:.2f}, TDM: {tdm:.2f} ")


def eval_forecasts(y, yhat):
    n = y.shape[0]
    horizon = y.shape[1]

    stats = pd.DataFrame(index=range(n), columns=["TDI", "TDM"])
    sq_err = pd.DataFrame(index=range(n), columns=range(horizon))
    abs_err = pd.DataFrame(index=range(n), columns=range(horizon))

    for i in range(n):
        tdi, tdm = calc_tdi_tdm(y[i], yhat[i])
        stats.loc[i, "TDI"] = tdi
        stats.loc[i, "TDM"] = tdm

        sq_err.loc[i] = squared_error(y[i], yhat[i])
        abs_err.loc[i] = absolute_error(y[i], yhat[i])

    return stats, sq_err, abs_err
