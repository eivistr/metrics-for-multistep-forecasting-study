import numpy as np
import pmdarima as pm
from tqdm import tqdm


MAX_ITER = 50


def get_forecasts(model, test, in_size, out_size):

    x, y, yhat = [], [], []

    # Update model on first input size sample
    x_i = test[0:in_size]
    y_i = test[in_size:in_size + out_size]

    model.update(x_i)
    yhat_i = model.predict(n_periods=out_size)  # Predict n steps into the future

    x.append(x_i)
    y.append(y_i)
    yhat.append(yhat_i)

    # Iterate from the first window, updating with the next step
    for i in tqdm(range(1, len(test) - in_size - out_size), unit="step", desc=f"Generating ARIMA forecasts"):
        x_i = test[i:i + in_size]
        y_i = test[i + in_size:i + in_size + out_size]

        model.update(x_i[-1])  # Update with LAST value as stepsize is 1 and the model retains previous values
        yhat_i = model.predict(n_periods=out_size)

        x.append(x_i)
        y.append(y_i)
        yhat.append(yhat_i)

    x, y, yhat = np.array(x), np.array(y), np.array(yhat)
    assert x.shape == y.shape == yhat.shape, "Forecast outputs must have equal dimensions"

    return x, y, yhat


def run_arima_model(train, test, in_size, out_size, arima_window, arima_m):

    assert train.ndim == test.ndim == 1, "Input arrays must be 1-dimentional"

    if arima_m > 1:
        print(f"Fitting SARIMA on {arima_window} values ...")
        model = pm.auto_arima(train[-arima_window:],
                              start_p=2, start_q=2, max_p=5, max_q=5,
                              seasonal=True, m=arima_m, max_P=5, max_Q=5,
                              trace=False,
                              suppress_warnings=True,
                              error_action='ignore',
                              stepwise=True,
                              maxiter=MAX_ITER)

    elif arima_m == 1:
        print(f"Fitting ARIMA on {arima_window} values ...")
        model = pm.auto_arima(train[-arima_window:],
                              start_p=2, start_q=2, max_p=5, max_q=5,
                              seasonal=False,
                              trace=False,
                              suppress_warnings=True,
                              error_action='ignore',
                              stepwise=True,
                              maxiter=MAX_ITER)
    else:
        raise ValueError("Parameter arima_m must be positive integer")

    # Get forecasts on test
    x_test, y_test, yhat_test = get_forecasts(model, test, in_size, out_size)

    return x_test, y_test, yhat_test



