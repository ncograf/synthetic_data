from typing import Any, Dict, List

import boosted_stats
import lagged_correlation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from power_fit import fit_powerlaw


def leverage_effect_torch(log_returns: torch.Tensor, max_lag: int) -> torch.Tensor:
    """Linear unpredictability

    :math:`Corr(r_t, r_{t+k}) \approx 0, \quad \text{for } k \geq 1`

    where k is chosen sucht that k <= `max_lag`

    Args:
        log_returns (pd.DataFrame): price log returns
        max_lag (int): maximal lag to compute

    Returns:
        npt.NDArray: max_lag x (log_returns.shape[1]) for each stock
    """

    if not torch.is_tensor(log_returns):
        log_returns = torch.tensor(log_returns)

    # make asolute values
    vol_returns = log_returns**2

    var = torch.nanmean((log_returns - torch.nanmean(log_returns, dim=0)) ** 2, dim=0)
    cov = lagged_correlation.lagged_corr(
        log_returns, vol_returns, max_lag=max_lag, dim=0
    )[1:]

    vol_cluster = cov / torch.pow(var, 3 / 2)
    return vol_cluster


def leverage_effect(log_returns: npt.ArrayLike, max_lag: int) -> npt.NDArray:
    """Leverage effect

    Computes the correlation between current returns and future volatility (squared returns)

    Args:
        log_returns (npt.ArrayLike): log returns
        max_lag (int): maximal lag

    Raises:
        RuntimeError: Wrong dimension

    Returns:
        ndarray: (max_lag x stocks) leverage effects for different lags
    """

    log_returns = np.asarray(log_returns)
    if log_returns.ndim == 1:
        log_returns = log_returns.reshape((-1, 1))
    elif log_returns.ndim > 2:
        raise RuntimeError(
            f"Log Returns have {log_returns.ndim} dimensions must have 1 or 2."
        )

    if log_returns.dtype.name == "float32":
        stat = boosted_stats.leverage_effect_float(log_returns, max_lag, False)
    elif log_returns.dtype.name == "float64":
        stat = boosted_stats.leverage_effect_double(log_returns, max_lag, False)

    return stat


def leverage_effect_stats(log_returns: npt.ArrayLike, max_lag: int) -> Dict[str, Any]:
    """Leverage effect statistics

    Args:
        log_returns (npt.ArrayLike): log returns
        max_lag (int): maximum lag

    Returns:
        Dict[str, Any]: result dictonary with keys:
            lev_eff: volatility clustering data (max_lag x stocks)
            corr: pearson correlation coefficient of powerlaw fit
            beta: exponent fitted in powerlaw
            alpha : constant fitted in powerlaw
            corr_std: standard deviation for negative fits
            beta_std: standard deviation for negative fits
            alpha_std: standard deviation for negative fits
            corr_max: max fit correlation over index
            corr_min: min fit correlation over index
            beta_min: min fit rate over index
            beta_max: max fit rate over index
            beta_mean: mean fit rate over index
            beta_median: median fit rate over index
    """

    if not torch.is_tensor(log_returns):
        log_returns = torch.tensor(log_returns)

    leveff = leverage_effect_torch(log_returns, max_lag=max_lag)
    leveff = np.asarray(leveff)
    x = np.arange(1, leveff.shape[0] + 1)
    y = np.mean(leveff, axis=1)
    _, _, alpha, beta, corr = fit_powerlaw(x, -y, optimize="none")

    # variace estimation
    alpha_arr, beta_arr, corr_arr = [], [], []
    for idx in range(leveff.shape[1]):
        _, _, pa, pb, pr = fit_powerlaw(x, -leveff[:, idx], optimize="none")
        alpha_arr.append(pa)
        beta_arr.append(pb)
        corr_arr.append(pr)

    stats = {
        "lev_eff": leveff,
        "corr": corr,
        "beta": beta,
        "alpha": alpha,
        "corr_std": np.nanstd(corr_arr),
        "alpha_std": np.nanstd(alpha_arr),
        "corr_max": np.nanmax(corr_arr),
        "corr_min": np.nanmin(corr_arr),
        "beta_std": np.nanstd(beta_arr),
        "beta_max": np.nanmax(beta_arr),
        "beta_min": np.nanmin(beta_arr),
        "beta_mean": np.nanmean(beta_arr),
        "beta_median": np.nanmedian(beta_arr),
    }

    return stats


def visualize_stat(
    plot: plt.Axes, log_returns: npt.NDArray, name: str, print_stats: List[str]
):
    stat = leverage_effect_stats(log_returns=log_returns, max_lag=100)
    lev_eff = stat["lev_eff"]
    y = np.mean(lev_eff, axis=1)
    x = np.arange(1, y.size + 1)

    pow_c, pow_rate = (
        stat["alpha"],
        stat["beta"],
    )
    x_lin = np.linspace(np.min(x), np.max(x), num=100)
    # y_lin = -exp_c * np.exp(-x_lin / exp_tau)
    # y_lin_qiu = -20 * np.exp(-x_lin / 12)
    y_pow = -np.exp(pow_c) * np.power(x_lin, pow_rate)

    lev_eff_axes_setting = {
        "title": f"{name} leverage effect",
        "ylabel": r"$L(k)$",
        "xlabel": "lag k",
        "xscale": "linear",
        "yscale": "linear",
    }
    lev_eff_plot_setting = {
        "alpha": 0.8,
        "marker": "None",
        "color": "cornflowerblue",
        "markersize": 0,
        "linestyle": "-",
        "linewidth": 2,
    }

    plot.set(**lev_eff_axes_setting)
    plot.plot(x, y, **lev_eff_plot_setting)
    # plot.plot(x_lin, y_lin, label="$\\tau = 50.267$", color="red", linestyle="--", alpha=1)
    plot.plot(
        x_lin[2:],
        y_pow[2:],
        label=f"$L(k) \propto  k^{{{pow_rate:.2f}}}$",
        color="navy",
        linestyle="--",
        linewidth=2,
        alpha=1,
    )
    # plot.plot( x_lin[5:], y_lin_qiu[5:], label="Qiu paper $\\tau = 13$", color="navy", linestyle="--", alpha=0.3)
    plot.axhline(y=0, linestyle="--", c="black", alpha=0.4)

    for key in print_stats:
        print(f"{name} leverage effect {key} {stat[key]}")

    plot.legend(loc="lower right")
