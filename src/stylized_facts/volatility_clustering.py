from typing import Any, Dict, List

import boosted_stats
import lagged_correlation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from power_fit import fit_powerlaw


def volatility_clustering_torch(
    log_returns: torch.Tensor, max_lag: int
) -> torch.Tensor:
    """Compute Autocorrelation of absolute log returns

    Args:
        log_returns (array_like): log_returns
        max_lag (int): maximum lag

    Returns:
        npt.NDArray: autocorrelation (max_lag x stocks)
    """

    if not torch.is_tensor(log_returns):
        log_returns = torch.tensor(log_returns)

    # make asolute values
    log_returns = torch.abs(log_returns)

    # compute the means / var for each stock
    var = torch.nanmean((log_returns - torch.nanmean(log_returns, dim=0)) ** 2, dim=0)

    # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
    cov = lagged_correlation.auto_corr(log_returns, max_lag=max_lag, dim=0)[1:]

    vol_cluster = cov / var

    return vol_cluster


def volatility_clustering(log_returns: npt.ArrayLike, max_lag: int) -> npt.NDArray:
    """Compute Autocorrelation of absolute log returns

    Args:
        log_returns (array_like): log_returns
        max_lag (int): maximum lag

    Returns:
        npt.NDArray: autocorrelation (max_lag x stocks)
    """

    log_returns = np.asarray(log_returns)
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
    centered_abs_log_returns = abs_log_returns - mu
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

    vol_clust = volatility_clustering_torch(
        log_returns=torch.tensor(log_returns), max_lag=max_lag
    )
    vol_clust = np.asarray(vol_clust)
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

    stats = {
        "vol_clust": vol_clust,
        "power_fit_x": fit_x,
        "corr": corr,
        "beta": beta,
        "alpha": alpha,
        "corr_std": np.nanstd(r_arr),
        "beta_std": np.nanstd(beta_arr),
        "alpha_std": np.nanstd(alpha_arr),
        "corr_min": np.nanmin(r_arr),
        "corr_max": np.nanmax(r_arr),
        "beta_max": np.nanmax(beta_arr),
        "beta_min": np.nanmin(beta_arr),
        "beta_mean": np.nanmean(beta_arr),
        "beta_median": np.nanmedian(beta_arr),
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
