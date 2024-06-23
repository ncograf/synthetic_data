from typing import Tuple

import numpy as np
import numpy.typing as npt


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
powerlaw_pos_style = {
    "alpha": 0.5,
    "marker": "none",
    "color": "blue",
    "markersize": 1,
    "linestyle": "--",
    "label": r"powerlaw fit $P\left( \tilde{r_t} \right) \approx cx^\alpha$",
}
powerlaw_neg_style = {
    "alpha": 0.5,
    "marker": "none",
    "color": "red",
    "markersize": 1,
    "linestyle": "--",
    "label": r"powerlaw fit $P\left( \tilde{r_t} \right) \approx cx^\alpha$",
}
