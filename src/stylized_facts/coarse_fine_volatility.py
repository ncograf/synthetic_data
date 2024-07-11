from typing import Any, Dict, List, Tuple

import boosted_stats
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from power_fit import fit_powerlaw


def coarse_fine_volatility(
    log_returns: npt.ArrayLike, tau: int, max_lag: int
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Coarse fine volatility: compute the lead lag

    :math:```
        p(k) = Corr(v_c(t+k), v_f)
        v_c(t) = | \sum_i r_{t - i} |
        v_f(t) = \sum_i | r_{t - i} |

        \Delta p(k) = p(k) - p(-k)
        ```
    Args:
        log_returns (npt.ArrayLike): returns
        tau (int): timespan to sum over in v_c and v_f
        max_lag (int): max lag to compute in correlation

    Returns:
        Tuple[ndarray, ndarray, ndarray, ndarray]: lead_lag p(k), lead_lag x, delta lead lag, delta lead lag x
    """

    log_returns = np.asarray(log_returns)
    if log_returns.ndim == 1:
        log_returns = log_returns.reshape((-1, 1))
    elif log_returns.ndim > 2:
        raise RuntimeError(
            f"Log Returns have {log_returns.ndim} dimensions must have 1 or 2."
        )

    # hack by setting nans to 0
    nan_mask = np.isnan(log_returns)
    log_returns[nan_mask] = 0

    nan_mask_shifted = np.isnan(log_returns[:-tau])

    # compute coarse volatiltiy |sum r_t|
    cum_sum = np.cumsum(log_returns, axis=0)
    v_c_tau = np.abs(cum_sum[tau:] - cum_sum[:-tau])
    v_c_tau[nan_mask_shifted] = np.nan

    # compute fine volatiltiy sum |r_t|
    abs_log_return = np.abs(log_returns)
    abs_cumsum = np.cumsum(abs_log_return, axis=0)
    v_f_tau = abs_cumsum[tau:] - abs_cumsum[:-tau]
    v_f_tau = v_f_tau / tau
    v_f_tau[nan_mask_shifted] = np.nan

    # center variables
    v_c_mean = np.nanmean(v_c_tau, axis=0)
    v_c_tau_centered = v_c_tau - v_c_mean

    v_f_mean = np.nanmean(v_f_tau, axis=0)
    v_f_tau_centered = v_f_tau - v_f_mean

    # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
    if log_returns.dtype.name == "float32":
        stat_pos = boosted_stats.lag_prod_mean_float(
            v_c_tau_centered, v_f_tau_centered, max_lag, False
        )
        stat_neg = boosted_stats.lag_prod_mean_float(
            v_f_tau_centered, v_c_tau_centered, max_lag, False
        )
    elif log_returns.dtype.name == "float64":
        stat_pos = boosted_stats.lag_prod_mean_double(
            v_c_tau_centered, v_f_tau_centered, max_lag, False
        )
        stat_neg = boosted_stats.lag_prod_mean_double(
            v_f_tau_centered, v_c_tau_centered, max_lag, False
        )

    # normalize to get correlation
    v_f_std = np.nanstd(v_f_tau, axis=0)
    v_c_std = np.nanstd(v_c_tau, axis=0)
    stat_neg = stat_neg / (v_c_std * v_f_std)
    stat_pos = stat_pos / (v_c_std * v_f_std)

    stat = np.concatenate([np.flip(stat_neg, axis=0), stat_pos[1:]], axis=0)
    lead_lag = stat_pos - stat_neg

    lead_lag_x = np.arange(max_lag + 1)
    stat_x = np.concatenate([-np.flip(lead_lag_x), lead_lag_x[1:]])

    return stat, stat_x, lead_lag, lead_lag_x


def coarse_fine_volatility_stats(
    log_returns: npt.ArrayLike, tau: int, max_lag: int
) -> Dict[str, Any]:
    """Coarse Fine Volatility Statistics

    :math:```
        p(k) = Corr(v_c(t+k), v_f)
        v_c(t) = | \sum_i r_{t - i} |
        v_f(t) = \sum_i | r_{t - i} |

        \Delta p(k) = p(k) - p(-k)
        ```

    Args:
        log_returns (npt.ArrayLike): log returns
        tau (int): timespan to sum over in v_c and v_f
        max_lag (int): maximum lag

    Returns:
        Dict[str, Any]: result dictonary with keys:
            lead_lag: lead lag p(k) as defined above
            lead_lag_k: values for k in lead_lag
            delta_lead_lag: delta lead lag as defined above
            delta_lead_lag_k: values for k in delta lead lag
            argmin : argmin delta lead lag
            beta : rate for fit in delta lead lag (from argmin to last)
            alpha : constant for fit in delta lead lag (from argmin to last)
            coor : pearson correlation for fit
            argmin_std : standard deviation argmin delta lead lag
            beta_std: standard deviation for fits
            alpha_std: standard deviation for fits
            corr_std: standard deviation for fits
    """

    ll, ll_x, dll, dll_x = coarse_fine_volatility(
        log_returns=log_returns, tau=tau, max_lag=max_lag
    )

    argmin = np.nanargmin(np.nanmean(dll, axis=1))
    _, _, alpha, beta, corr = fit_powerlaw(
        dll_x[argmin:], -np.nanmean(dll[argmin:], axis=1), optimize="none"
    )

    # variace estimation
    alpha_arr, beta_arr, r_arr, amin_arr = [], [], [], []
    for idx in range(dll.shape[1]):
        amin = np.nanargmin(dll[:, idx])
        _, _, a, b, r = fit_powerlaw(dll_x[amin:], -dll[amin:, idx], optimize="none")
        amin_arr.append(amin)
        alpha_arr.append(a)
        beta_arr.append(b)
        r_arr.append(r)

    stats = {
        "lead_lag": ll,
        "lead_lag_k": ll_x,
        "delta_lead_lag": dll,
        "delta_lead_lag_k": dll_x,
        "argmin": argmin,
        "beta": beta,
        "alpha": alpha,
        "corr": corr,
        "argmin_std": np.nanstd(amin_arr),
        "alpha_std": np.nanstd(alpha_arr),
        "beta_std": np.nanstd(beta_arr),
        "corr_std": np.nanstd(r_arr),
        "beta_min": np.nanmin(beta_arr),
        "beta_max": np.nanmax(beta_arr),
        "beta_median": np.nanmedian(beta_arr),
        "beta_mean": np.nanmean(beta_arr),
    }

    return stats


def visualize_stat(
    plot: plt.Axes, log_returns: npt.NDArray, name: str, print_stats: List[str]
):
    stat = coarse_fine_volatility_stats(log_returns=log_returns, tau=5, max_lag=100)
    dll, dll_x = np.mean(stat["delta_lead_lag"], axis=1), stat["delta_lead_lag_k"]
    ll, ll_x = np.mean(stat["lead_lag"], axis=1), stat["lead_lag_k"]
    argmin, alpha, beta = stat["argmin"], stat["alpha"], stat["beta"]

    cf_vol_axes_setting = {
        "title": f"{name} coarse-fine volatility",
        "ylabel": r"$\rho(k)$",
        "xlabel": "lag(k days)",
        "xscale": "linear",
        "yscale": "linear",
    }
    cf_vol_plot_setting = {
        "alpha": 1,
        "marker": "o",
        "color": "cornflowerblue",
        "markersize": 2,
        "linestyle": "None",
        "label": r"$\rho(k)$",
    }
    lead_lag_plot_setting = {
        "alpha": 1,
        "marker": "None",
        "color": "orange",
        "markersize": 0,
        "linestyle": "-",
        "label": r"$\Delta \rho(k)$",
    }
    plot.set(**cf_vol_axes_setting)
    plot.plot(dll_x, dll, c="blue")
    plot.plot(
        dll_x[argmin],
        dll[argmin],
        linestyle="none",
        marker="o",
        markersize=5,
        c="red",
    )
    plot.plot(ll_x, ll, **cf_vol_plot_setting)
    plot.plot(dll_x, dll, **lead_lag_plot_setting)
    if np.abs(beta) < 1e-6:
        print(beta)
    x_lin = dll_x[argmin + 2 :]
    y_lin = -np.exp(alpha) * (x_lin**beta)
    plot.plot(
        x_lin,
        y_lin,
        label=f"$\\Delta \\rho \\propto k^{{{beta:.2f}}} $",
        linestyle="--",
        color="red",
        alpha=1,
    )
    plot.axhline(y=0, linestyle="--", c="black", alpha=0.4)

    for key in print_stats:
        print(f"{name} coarse fine volatility {key} {stat[key]}")

    plot.legend(loc="upper right")
