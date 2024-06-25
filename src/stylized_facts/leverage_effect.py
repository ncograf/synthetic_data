from typing import Any, Dict

import boosted_stats
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


lev_eff_axes_setting = {
    "title": "leverage effect",
    "ylabel": r"$L(k)$",
    "xlabel": "lag k",
    "xscale": "linear",
    "yscale": "linear",
}
lev_eff_plot_setting = {
    "alpha": 1,
    "marker": "None",
    "color": "royalblue",
    "markersize": 0,
    "linestyle": "-",
}
