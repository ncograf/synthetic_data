from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from power_fit import fit_powerlaw


def _discrete_pdf(log_returns: npt.ArrayLike, n_bins: int = 1000):
    # distribution of all returns
    log_returns = np.asarray(log_returns)
    log_returns = log_returns[~np.isnan(log_returns)]

    # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
    bin_density, bin_edges = np.histogram(log_returns, bins=n_bins, density=False)
    bin_starts = bin_edges[:-1]
    bin_density = bin_density / log_returns.size
    return bin_density, bin_starts


def discrete_pdf(log_returns: npt.ArrayLike, n_bins: int = 1000) -> Tuple[npt.NDArray]:
    """Compute `heavy tails`, i.e. pdf approximation as discrete histogram

    Args:
        log_returns (array_like): data to be fit
        n_bins (int, optional): number of bins in histogram. Defaults to 1000.

    Returns:
        Tuple[npt.NDArray]: positive_density, positive_bins, negative_density, negative_bins
    """
    log_returns = np.asarray(log_returns)
    if log_returns.ndim == 1:
        log_returns = log_returns.reshape((-1, 1))
    elif log_returns.ndim > 2:
        raise RuntimeError(
            f"Log Returns have {log_returns.ndim} dimensions must have 1 or 2."
        )

    # normalize returns
    std = np.nanstd(log_returns)
    log_returns = log_returns / std

    dens, bins = _discrete_pdf(log_returns, n_bins=2 * n_bins)

    pos_dens = dens[bins > 0]
    pos_bins = bins[bins > 0]
    neg_dens = np.flip(dens[bins < 0])
    neg_bins = -np.flip(bins[bins < 0])

    return pos_dens, pos_bins, neg_dens, neg_bins


def heavy_tails_stats(
    log_returns: npt.ArrayLike, n_bins: int, tail_quant: float
) -> Dict[str, Any]:
    """Heavy tails statistics

    Args:
        log_returns (npt.ArrayLike): log returns
        n_bins (int): number of bins for histogram
        tail_quant (float): quantile of the tails to be considered in the fit

    Returns:
        Dict[str, Any]: result dictonary with keys:
            pos_dens: heavy_tails data positive probabilities (n_bins x stocks)
            pos_bins: heavy_tails data positive bins (n_bins x stocks)
            pos_powerlaw_x: positive powerlaw x values used for fit
            pos_corr: pearson correlation coefficient of powerlaw fit
            pos_beta: exponent fitted in powerlaw
            pos_alpha : constant fitted in powerlaw
            pos_beta_std: standard deviation for positive fits
            pos_alpha_std: standard deviation for positive fits
            pos_corr_std: standard deviation for positive fits
            neg_dens: heavy_tails data negative probabilities (n_bins x stocks)
            neg_bins: heavy_tails data negative bins (n_bins x stocks)
            neg_powerlaw_x: negative powerlaw x values used for fit
            neg_corr: pearson correlation coefficient of powerlaw fit
            neg_beta: exponent fitted in powerlaw
            neg_alpha : constant fitted in powerlaw
            neg_std: mean squared error standard deviation over stocks
            neg_beta_std: standard deviation for negative fits
            neg_alpha_std: standard deviation for negative fits
            neg_corr_std: standard deviation for negative fits
    """

    pos_y, pos_x, neg_y, neg_x = discrete_pdf(log_returns=log_returns, n_bins=n_bins)
    pos_mask, neg_mask = np.cumsum(pos_y), np.cumsum(neg_y)
    pos_mask, neg_mask = (
        pos_mask > pos_mask[-1] * (1 - tail_quant) if pos_mask.size > 0 else [],
        neg_mask > neg_mask[-1] * (1 - tail_quant) if neg_mask.size > 0 else [],
    )

    pos_fit_x, _, pos_alpha, pos_beta, pos_r = fit_powerlaw(
        pos_x[pos_mask], pos_y[pos_mask], optimize="none"
    )
    neg_fit_x, _, neg_alpha, neg_beta, neg_r = fit_powerlaw(
        neg_x[neg_mask], neg_y[neg_mask], optimize="none"
    )

    # variace estimation
    pos_y_arr, pos_x_arr, neg_y_arr, neg_x_arr = [], [], [], []
    for i in range(log_returns.shape[1]):
        py, px, ny, nx = discrete_pdf(log_returns[:, i], n_bins // 40)
        pos_mask, neg_mask = np.cumsum(py), np.cumsum(ny)
        pos_mask, neg_mask = (
            pos_mask > pos_mask[-1] * (1 - tail_quant) if pos_mask.size > 0 else [],
            neg_mask > neg_mask[-1] * (1 - tail_quant) if neg_mask.size > 0 else [],
        )
        pos_y_arr.append(py[pos_mask])
        pos_x_arr.append(px[pos_mask])
        neg_y_arr.append(ny[neg_mask])
        neg_x_arr.append(nx[neg_mask])

    pos_alpha_arr, pos_beta_arr, pos_r_arr = [], [], []
    for px, py in zip(pos_x_arr, pos_y_arr):
        _, _, a, b, r = fit_powerlaw(px, py, optimize="none")
        pos_alpha_arr.append(a)
        pos_beta_arr.append(b)
        pos_r_arr.append(r)

    neg_alpha_arr, neg_beta_arr, neg_r_arr = [], [], []
    for px, py in zip(neg_x_arr, neg_y_arr):
        _, _, a, b, r = fit_powerlaw(px, py, optimize="none")
        neg_alpha_arr.append(a)
        neg_beta_arr.append(b)
        neg_r_arr.append(r)

    stats = {
        "pos_dens": pos_y,
        "pos_bins": pos_x,
        "pos_powerlaw_x": pos_fit_x,
        "pos_corr": pos_r,
        "pos_beta": pos_beta,
        "pos_alpha": pos_alpha,
        "pos_alpha_std": np.nanstd(pos_alpha_arr),
        "pos_corr_std": np.nanstd(pos_r_arr),
        "pos_beta_std": np.nanstd(pos_beta_arr),
        "pos_beta_max": np.nanmax(pos_beta_arr),
        "pos_beta_min": np.nanmin(pos_beta_arr),
        "pos_beta_mean": np.nanmean(pos_beta_arr),
        "pos_beta_median": np.nanmedian(pos_beta_arr),
        "neg_dens": neg_y,
        "neg_bins": neg_x,
        "neg_powerlaw_x": neg_fit_x,
        "neg_corr": neg_r,
        "neg_beta": neg_beta,
        "neg_alpha": neg_alpha,
        "neg_alpha_std": np.nanstd(neg_alpha_arr),
        "neg_corr_std": np.nanstd(neg_r_arr),
        "neg_beta_std": np.nanstd(neg_beta_arr),
        "neg_beta_max": np.nanmax(neg_beta_arr),
        "neg_beta_min": np.nanmin(neg_beta_arr),
        "neg_beta_mean": np.nanmean(neg_beta_arr),
        "neg_beta_median": np.nanmedian(neg_beta_arr),
    }

    return stats


def visualize_stat(
    plot: plt.Axes, log_returns: npt.NDArray, name: str, print_stats: List[str]
):
    # compute statistics
    stat = heavy_tails_stats(log_returns=log_returns, n_bins=1000, tail_quant=0.1)
    pos_x, pos_y, pos_fit_x = stat["pos_bins"], stat["pos_dens"], stat["pos_powerlaw_x"]
    neg_x, neg_y, neg_fit_x = stat["neg_bins"], stat["neg_dens"], stat["neg_powerlaw_x"]

    # plot data
    plot.set(
        title=f"{name} heavy tails",
        ylabel=r"density $P\left(\tilde{r_t}\right)$",
        xlabel=r"normalized returns $\tilde{r_t} := r_t\, /\, \sigma$",
        xscale="log",
        yscale="log",
    )
    plot.plot(
        pos_x,
        pos_y,
        alpha=0.8,
        marker="o",
        color="violet",
        markersize=2,
        linestyle="None",
        label=r"pos. $\tilde{r}_t > 0$",
    )
    plot.plot(
        neg_x,
        neg_y,
        alpha=0.8,
        marker="o",
        color="cornflowerblue",
        markersize=2,
        linestyle="None",
        label=r"neg. $\tilde{r}_t < 0$",
    )
    xlim = plot.get_xlim()
    ylim = plot.get_ylim()

    # compute positive fits
    pos_alpha, pos_beta = stat["pos_alpha"], stat["pos_beta"]
    pos_x_lin = np.linspace(np.min(pos_fit_x), np.max(pos_fit_x), num=1000)
    pos_y_lin = np.exp(pos_alpha) * np.power(pos_x_lin, pos_beta)

    # adjust the lines to fit the plot
    pos_filter = (pos_y_lin > ylim[0]) & (pos_y_lin < ylim[1])
    pos_x_lin = pos_x_lin[pos_filter][:-100]
    pos_y_lin = pos_y_lin[pos_filter][:-100]

    # compute negative fit
    neg_alpha, neg_beta = stat["neg_alpha"], stat["neg_beta"]
    neg_x_lin = np.linspace(np.min(neg_fit_x), np.max(neg_fit_x), num=1000)
    neg_y_lin = np.exp(neg_alpha) * np.power(neg_x_lin, neg_beta)

    # adjust the lines to fit the plot
    neg_filter = (neg_y_lin > ylim[0]) & (neg_y_lin < ylim[1])
    neg_x_lin = neg_x_lin[neg_filter][:-100]
    neg_y_lin = neg_y_lin[neg_filter][:-100]

    # print statistics if needed
    for key in print_stats:
        print(f"{name} heavy tails {key} {stat[key]}")

    # plot the fitted lines
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
    plot.set_xlim(xlim)
    plot.set_ylim(ylim)
    plot.legend(loc="lower left")
