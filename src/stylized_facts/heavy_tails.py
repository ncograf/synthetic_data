from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from power_fit import fit_powerlaw


def _heavy_tails(log_returns: npt.ArrayLike, n_bins: int = 1000):
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


def heavy_tails(log_returns: npt.ArrayLike, n_bins: int = 1000) -> Tuple[npt.NDArray]:
    """Compute `heavy tails`, i.e. pdf approximation as discrete histogram

    Args:
        log_returns (array_like): data to be fit
        n_bins (int, optional): number of bins in histogram. Defaults to 1000.

    Returns:
        Tuple[npt.NDArray]: positive_density, positive_bins, negative_density, negative_bins
    """
    log_returns = np.array(log_returns)
    if log_returns.ndim == 1:
        log_returns = log_returns.reshape((-1, 1))
    elif log_returns.ndim > 2:
        raise RuntimeError(
            f"Log Returns have {log_returns.ndim} dimensions must have 1 or 2."
        )

    pos_log_returns = log_returns[log_returns > 0]
    neg_log_returns = log_returns[log_returns < 0]

    neg_dens, neg_bins = _heavy_tails(-neg_log_returns, n_bins=n_bins)
    pos_dens, pos_bins = _heavy_tails(pos_log_returns, n_bins=n_bins)

    return pos_dens, pos_bins, neg_dens, neg_bins


def heavy_tails_stats(log_returns: npt.ArrayLike, n_bins: int) -> Dict[str, Any]:
    """Heavy tails statistics

    Args:
        log_returns (npt.ArrayLike): log returns
        n_bins (int): number of bins for histogram

    Returns:
        Dict[str, Any]: result dictonary with keys:
            pos_dens: heavy_tails data positive probabilities (n_bins x stocks)
            pos_bins: heavy_tails data positive bins (n_bins x stocks)
            pos_powerlaw_x: positive powerlaw x values used for fit
            pos_corr: pearson correlation coefficient of powerlaw fit
            pos_rate: exponent fitted in powerlaw
            pos_const : constant fitted in powerlaw
            pos_rate_std: standard deviation for positive fits
            pos_const_std: standard deviation for positive fits
            pos_corr_std: standard deviation for positive fits
            neg_dens: heavy_tails data negative probabilities (n_bins x stocks)
            neg_bins: heavy_tails data negative bins (n_bins x stocks)
            neg_powerlaw_x: negative powerlaw x values used for fit
            neg_corr: pearson correlation coefficient of powerlaw fit
            neg_rate: exponent fitted in powerlaw
            neg_const : constant fitted in powerlaw
            neg_std: mean squared error standard deviation over stocks
            neg_rate_std: standard deviation for negative fits
            neg_const_std: standard deviation for negative fits
            neg_corr_std: standard deviation for negative fits
    """

    pos_y, pos_x, neg_y, neg_x = heavy_tails(log_returns=log_returns, n_bins=n_bins)
    pos_fit_x, _, pos_alpha, pos_beta, pos_r = fit_powerlaw(
        pos_x, pos_y, optimize="both"
    )
    neg_fit_x, _, neg_alpha, neg_beta, neg_r = fit_powerlaw(
        neg_x, neg_y, optimize="both"
    )

    # variace estimation
    pos_y_arr, pos_x_arr, neg_y_arr, neg_x_arr = [], [], [], []
    for i in range(log_returns.shape[1]):
        py, px, ny, nx = heavy_tails(log_returns[:, i], n_bins)
        pos_y_arr.append(py)
        pos_x_arr.append(px)
        neg_y_arr.append(ny)
        neg_x_arr.append(nx)

    pos_alpha_arr, pos_beta_arr, pos_r_arr = [], [], []
    for px, py in zip(pos_x_arr, pos_y_arr):
        _, _, a, b, r = fit_powerlaw(px, py, optimize="both")
        pos_alpha_arr.append(a)
        pos_beta_arr.append(b)
        pos_r_arr.append(r)

    neg_alpha_arr, neg_beta_arr, neg_r_arr = [], [], []
    for px, py in zip(neg_x_arr, neg_y_arr):
        _, _, a, b, r = fit_powerlaw(px, py, optimize="both")
        neg_alpha_arr.append(a)
        neg_beta_arr.append(b)
        neg_r_arr.append(r)

    pos_std_alpha = np.std(pos_alpha_arr)
    pos_std_beta = np.std(pos_beta_arr)
    pos_std_r = np.std(pos_r_arr)

    neg_std_alpha = np.std(neg_alpha_arr)
    neg_std_beta = np.std(neg_beta_arr)
    neg_std_r = np.std(neg_r_arr)

    stats = {
        "pos_dens": pos_y,
        "pos_bins": pos_x,
        "pos_powerlaw_x": pos_fit_x,
        "pos_corr": pos_r,
        "pos_rate": pos_beta,
        "pos_const": pos_alpha,
        "pos_rate_std": pos_std_beta,
        "pos_const_std": pos_std_alpha,
        "pos_corr_std": pos_std_r,
        "neg_dens": neg_y,
        "neg_bins": neg_x,
        "neg_powerlaw_x": neg_fit_x,
        "neg_corr": neg_r,
        "neg_rate": neg_beta,
        "neg_const": neg_alpha,
        "neg_rate_std": neg_std_beta,
        "neg_const_std": neg_std_alpha,
        "neg_corr_std": neg_std_r,
    }

    return stats


def visualize_stat(
    plot: plt.Axes, log_returns: npt.NDArray, name: str, print_stats: List[str]
):
    stat = heavy_tails_stats(log_returns=log_returns, n_bins=1000)
    pos_x, pos_y, pos_fit_x = stat["pos_bins"], stat["pos_dens"], stat["pos_powerlaw_x"]
    pos_alpha, pos_beta = stat["pos_const"], stat["pos_rate"]

    pos_x_lin = np.linspace(np.min(pos_fit_x), np.max(pos_fit_x), num=1000)
    pos_y_lin = np.exp(pos_alpha) * np.power(pos_x_lin, pos_beta)

    neg_x, neg_y, neg_fit_x = stat["neg_bins"], stat["neg_dens"], stat["neg_powerlaw_x"]
    neg_alpha, neg_beta = stat["neg_const"], stat["neg_rate"]

    neg_x_lin = np.linspace(np.min(neg_fit_x), np.max(neg_fit_x), num=1000)
    neg_y_lin = np.exp(neg_alpha) * np.power(neg_x_lin, neg_beta)

    for key in print_stats:
        print(f"{name} heavy tails {key} {stat[key]}")

    heavy_tail_axes_setting = {
        "title": f"{name} heavy tails",
        "ylabel": r"density $P\left(\tilde{r_t}\right)$",
        "xlabel": r"normalized return $\tilde{r_t} := \frac{r_t}{\sigma}$",
        "xscale": "log",
        "yscale": "log",
    }
    heavy_tail_neg_plot_setting = {
        "alpha": 0.8,
        "marker": "o",
        "color": "cornflowerblue",
        "markersize": 2,
        "linestyle": "None",
        "label": r"neg. $\tilde{r}_t < 0$",
    }
    heavy_tail_pos_plot_setting = {
        "alpha": 0.8,
        "marker": "o",
        "color": "violet",
        "markersize": 2,
        "linestyle": "None",
        "label": r"pos. $\tilde{r}_t > 0$",
    }

    plot.set(**heavy_tail_axes_setting)
    plot.plot(pos_x, pos_y, **heavy_tail_pos_plot_setting)
    plot.plot(neg_x, neg_y, **heavy_tail_neg_plot_setting)
    plot.plot(
        pos_x_lin,
        pos_y_lin,
        label=f"pos. $p(\\tilde{{r}}_t) \\propto \\tilde{{r}}_t^{{{pos_beta:.2f}}}$",
        linewidth=2,
        linestyle="--",
        alpha=1,
        color="red",
    )
    plot.plot(
        neg_x_lin,
        neg_y_lin,
        label=f"neg. $p(\\tilde{{r}}_t) \\propto \\tilde{{r}}_t^{{{neg_beta:.2f}}}$",
        linewidth=2,
        linestyle="--",
        alpha=1,
        color="navy",
    )
    plot.legend(loc="best")
