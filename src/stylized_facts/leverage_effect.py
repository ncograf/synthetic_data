from typing import Any, Dict, List

import boosted_stats
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from power_fit import fit_lin_log, fit_powerlaw


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

    log_returns = np.array(log_returns)
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
            exp_r: pearson correlation coefficient of powerlaw fit
            exp_tau: exponent fitted in exponential fit
            exp_c : constant fitted in expnetial fit
            exp_r_std: standard deviation in exponnetial fits
            exp_tau_std: standard deviation for exponential fits
            exp_r_std: standard deviation for exponential fits
            pow_r: pearson correlation coefficient of powerlaw fit
            pow_rate: exponent fitted in powerlaw
            pow_const : constant fitted in powerlaw
            pow_r_std: standard deviation for negative fits
            pow_rate_std: standard deviation for negative fits
            powe_const_std: standard deviation for negative fits
    """

    leveff = leverage_effect(log_returns=log_returns, max_lag=max_lag)
    x = np.arange(1, leveff.shape[0] + 1)
    y = np.mean(leveff, axis=1)
    _, _, pow_const, pow_rate, pow_r = fit_powerlaw(x, -y, optimize="none")
    exp_c, exp_tau, exp_r = fit_lin_log(x, -y)
    exp_tau = -1 / exp_tau
    exp_c = np.exp(exp_c)

    # variace estimation
    pow_const_arr, pow_rate_arr, pow_r_arr = [], [], []
    exp_c_arr, exp_tau_arr, exp_r_arr = [], [], []
    for idx in range(leveff.shape[1]):
        _, _, pa, pb, pr = fit_powerlaw(x, -leveff[:, idx], optimize="none")
        ec, et, er = fit_lin_log(x, -leveff[:, idx])
        pow_const_arr.append(pa)
        pow_rate_arr.append(pb)
        pow_r_arr.append(pr)

        exp_c_arr.append(np.exp(ec))
        exp_tau_arr.append(-1 / et)
        exp_r_arr.append(er)

    std_c_pow = np.std(pow_const_arr)
    std_rate_pow = np.std(pow_rate_arr)
    std_r_pow = np.std(pow_r_arr)

    std_c_exp = np.std(exp_c_arr)
    std_tau_exp = np.std(exp_tau_arr)
    std_r_exp = np.std(exp_r_arr)

    stats = {
        "lev_eff": leveff,
        "exp_r": exp_r,
        "exp_c": exp_c,
        "exp_tau": exp_tau,
        "exp_c_std": std_c_exp,
        "exp_tau_std": std_tau_exp,
        "exp_r_std": std_r_exp,
        "pow_r": pow_r,
        "pow_rate": pow_rate,
        "pow_const": pow_const,
        "pow_r_std": std_r_pow,
        "pow_rate_std": std_rate_pow,
        "pow_const_std": std_c_pow,
    }

    return stats


def visualize_stat(
    plot: plt.Axes, log_returns: npt.NDArray, name: str, print_stats: List[str]
):
    stat = leverage_effect_stats(log_returns=log_returns, max_lag=100)
    lev_eff = stat["lev_eff"]
    y = np.mean(lev_eff, axis=1)
    x = np.arange(1, y.size + 1)

    _, _, pow_c, pow_rate = (
        stat["exp_c"],
        stat["exp_tau"],
        stat["pow_const"],
        stat["pow_rate"],
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
        print(f"f{name} leverage effect {key} {stat[key]}")

    plot.legend(loc="best")
