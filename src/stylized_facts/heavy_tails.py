import numpy as np
import pandas as pd
from scipy.stats import linregress


def _heavy_tails(log_returns: pd.DataFrame, n_bins: int = 1000):
    log_returns = np.array(log_returns)

    # distribution of all returns
    log_returns = np.array(log_returns)
    log_returns = log_returns[~np.isnan(log_returns)]

    # normalize returns
    mu = np.nanmean(log_returns)
    std = np.nanstd(log_returns)
    normalized_log_returns = (log_returns - mu) / std

    # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
    bin_density, bin_edges = np.histogram(
        normalized_log_returns, bins=n_bins, density=False
    )
    bin_starts = bin_edges[:-1]
    bin_density = bin_density / log_returns.size
    return bin_density, bin_starts


def heavy_tails(log_returns: pd.DataFrame, n_bins: int = 1000):
    log_returns = np.array(log_returns)

    pos_log_returns = log_returns[log_returns > 0]
    neg_log_returns = log_returns[log_returns < 0]

    neg_dens, neg_bins = _heavy_tails(-neg_log_returns, n_bins=n_bins)
    pos_dens, pos_bins = _heavy_tails(pos_log_returns, n_bins=n_bins)

    return pos_dens, pos_bins, neg_dens, neg_bins


def fit_powerlaw(x, y):
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    log_x = np.log(x)
    log_y = np.log(y)

    fit = linregress(log_x, log_y)
    goodness = fit.rvalue
    slope = fit.slope
    intercept = fit.intercept

    return intercept, slope, goodness


heavy_tail_axes_setting = {
    "title": "heavy tails",
    "ylabel": r"density $P\left(\tilde{r_t}\right)$",
    "xlabel": r"normalized return $\tilde{r_t} := \frac{r_t}{\sigma}$",
    "xscale": "log",
    "yscale": "log",
}
heavy_tail_neg_plot_setting = {
    "alpha": 1,
    "marker": "o",
    "color": "red",
    "markersize": 1,
    "linestyle": "None",
    "label": r"negative $r_t < 0$",
}
heavy_tail_pos_plot_setting = {
    "alpha": 1,
    "marker": "o",
    "color": "blue",
    "markersize": 1,
    "linestyle": "None",
    "label": r"positive $r_t > 0$",
}
