from typing import Any, Dict, List

import boosted_stats
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from power_fit import fit_powerlaw


def volatility_clustering(log_returns: npt.ArrayLike, max_lag: int) -> npt.NDArray:
    """Compute Autocorrelation of absolute log returns

    Args:
        log_returns (array_like): log_returns
        max_lag (int): maximum lag

    Returns:
        npt.NDArray: autocorrelation (max_lag x stocks)
    """

    log_returns = np.array(log_returns)
    if log_returns.ndim == 1:
        log_returns = log_returns.reshape((-1, 1))
    elif log_returns.ndim > 2:
        raise RuntimeError(
            f"Log Returns have {log_returns.ndim} dimensions must have 1 or 2."
        )

    # make asolute values
    abs_log_returns = np.abs(log_returns)

    # compute the means / var for each stock
    mu = np.nanmean(abs_log_returns, axis=0)
    var = np.nanvar(abs_log_returns, axis=0)

    # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
    centered_abs_log_returns = np.array(abs_log_returns - mu)
    if centered_abs_log_returns.dtype.name == "float32":
        correlation = boosted_stats.lag_prod_mean_float(
            centered_abs_log_returns, max_lag, False
        )
    elif centered_abs_log_returns.dtype.name == "float64":
        correlation = boosted_stats.lag_prod_mean_double(
            centered_abs_log_returns, max_lag, False
        )

    vol_clustering = correlation / var
    return vol_clustering


def volatility_clustering_stats(
    log_returns: npt.ArrayLike, max_lag: int
) -> Dict[str, Any]:
    """Volatility clustering statistics

    Args:
        log_returns (npt.ArrayLike): log returns
        max_lag (int): maximum lag

    Returns:
        Dict[str, Any]: result dictonary with keys:
            vol_clust: volatility clustering data (max_lag x stocks)
            power_fit_x: powerlaw x values used for fit
            corr: pearson correlation coefficient of powerlaw fit
            beta: exponent fitted in powerlaw
            alpha : constant fitted in powerlaw
            corr_std: standard deviation for fits
            beta_std: standard deviation for fits
            alpha_std: standard deviation for fits
            corr_max: max correlation observed
            corr_min: min correlation observed
            beta_min: min value of the fitted rate over index
            beta_max: max value of the fitted rate over index
    """

    vol_clust = volatility_clustering(log_returns=log_returns, max_lag=max_lag)
    x = np.arange(1, vol_clust.shape[0] + 1)
    fit_x, _, alpha, beta, corr = fit_powerlaw(
        x, np.mean(vol_clust, axis=1), optimize=(0, 150)
    )

    # variace estimation
    alpha_arr, beta_arr, r_arr = [], [], []
    for idx in range(vol_clust.shape[1]):
        _, _, a, b, r = fit_powerlaw(x, vol_clust[:, idx], optimize=(0, 150))
        alpha_arr.append(a)
        beta_arr.append(b)
        r_arr.append(r)

    std_alpha = np.std(alpha_arr)
    std_beta = np.std(beta_arr)
    std_r = np.std(r_arr)
    r_max = np.max(r_arr)
    r_min = np.min(r_arr)
    beta_max = np.max(beta_arr)
    beta_min = np.min(beta_arr)

    stats = {
        "vol_clust": vol_clust,
        "power_fit_x": fit_x,
        "corr": corr,
        "beta": beta,
        "alpha": alpha,
        "corr_std": std_r,
        "beta_std": std_beta,
        "alpha_std": std_alpha,
        "corr_min": r_min,
        "corr_max": r_max,
        "beta_max": beta_max,
        "beta_min": beta_min,
        "beta_arr": beta_arr,
    }

    return stats


def visualize_stat(
    plot: plt.Axes, log_returns: npt.NDArray, name: str, print_stats: List[str]
):
    stat = volatility_clustering_stats(log_returns=log_returns, max_lag=1000)
    pos_y, x_lin, alpha, beta = (
        np.mean(stat["vol_clust"], axis=1),
        stat["power_fit_x"],
        stat["alpha"],
        stat["beta"],
    )
    pos_x = np.arange(1, pos_y.size + 1)
    y_lin = np.exp(alpha) * np.power(x_lin, beta)

    vol_clust_axes_setting = {
        "title": "volatility clustering",
        "ylabel": r"$Corr(|r_t|, |r_{t+k}|)$",
        "xlabel": "lag k",
        "xscale": "log",
        "yscale": "log",
    }
    vol_clust_plot_setting = {
        "alpha": 0.8,
        "marker": "o",
        "color": "cornflowerblue",
        "markersize": 2,
        "linestyle": "None",
    }

    plot.set(**vol_clust_axes_setting)
    plot.plot(pos_x, pos_y, **vol_clust_plot_setting)
    plot.plot(
        x_lin,
        y_lin,
        label=f"$Corr(|r_t|, |r_{{t+k}}|) \propto k^{{{beta:.2f}}}$",
        linestyle="--",
        linewidth=2,
        color="navy",
        alpha=1,
    )

    for key in print_stats:
        print(f"{name} vol cluster {key}: {stat[key]}")

    plot.legend(loc="lower left")

    return
