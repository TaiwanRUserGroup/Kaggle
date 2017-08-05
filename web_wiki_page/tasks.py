# -*- coding:utf8 -*-
from functools import partial
import pandas as pd
import numpy as np
from scipy.optimize import minimize


def SMAPE(y, y_hat):
    """
    Symmetric Mean Absolute Percentage Error
    """
    denom = np.abs(y) + np.abs(y_hat)
    err = np.abs(y-y_hat)/denom
    err[denom == 0] = 0
    return 200.0*np.nanmean(err)


def loss_smape(lags, y, coff):
    """
    lags: N by k matrix
    y: 1D array of size N
    coff: 1D array of size k+1, (weight, bias)
    """
    w, b = coff[:-1], coff[-1]
    y_hat = lags.dot(w) + b
    return SMAPE(y, y_hat)


def fit_model(num_lags,
              access,
              verbose=False,
              coff_init=None,
              method='Nelder-Mead'):
    """
    Fit a AR model for given time series
    """
    assert num_lags >= 1, \
        "num_lags must be at least 1, get {}".format(num_lags)
    win_size = num_lags+1
    X = []
    Y = []
    for index in range(0, len(access)-num_lags):
        window = access[index:index+win_size]
        lags, y = window[:-1], window[-1]
        if pd.isnull(y) or pd.isnull(lags).any():
            if verbose:
                print("skipping {}".format(series.index[index]))
            continue
        X.append(lags)
        Y.append(y)
    if len(X) == 0:
        return None
    X = np.array(X, dtype=np.float)
    Y = np.array(Y, dtype=np.float)
    loss = partial(loss_smape, X, Y)
    if coff_init is not None:
        coff = coff_init
    else:
        coff = np.random.rand(num_lags+1)
    fit_result = minimize(loss, coff, method=method)
    return fit_result
