from typing import Any, Dict, Tuple

import boosted_stats
import numpy as np
import numpy.typing as npt
from scipy.stats import linregress


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

    log_returns = np.array(log_returns)
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
            r : pearson correlation for fit
            argmin_std : standard deviation argmin delta lead lag
            beta_std: standard deviation for fits
            alpha_std: standard deviation for fits
            r_std: standard deviation for fits
    """

    ll, ll_x, dll, dll_x = coarse_fine_volatility(
        log_returns=log_returns, tau=tau, max_lag=max_lag
    )

    argmin = np.argmin(np.mean(dll, axis=1))
    fit = linregress(dll_x[argmin:], np.mean(dll[argmin:], axis=1))
    alpha, beta, corr = fit.intercept, fit.slope, fit.rvalue

    # variace estimation
    alpha_arr, beta_arr, r_arr, amin_arr = [], [], [], []
    for idx in range(dll.shape[1]):
        amin = np.argmin(dll[:, idx])
        fit = linregress(dll_x[amin:], dll[amin:, idx])
        a, b, r = fit.intercept, fit.slope, fit.rvalue
        amin_arr.append(amin)
        alpha_arr.append(a)
        beta_arr.append(b)
        r_arr.append(r)

    std_alpha = np.std(alpha_arr)
    std_beta = np.std(beta_arr)
    std_r = np.std(r_arr)
    std_amin = np.std(amin_arr)

    stats = {
        "lead_lag": ll,
        "lead_lag_k": ll_x,
        "delta_lead_lag": dll,
        "delta_lead_lag_k": dll_x,
        "argmin": argmin,
        "beta": beta,
        "alpha": alpha,
        "r": corr,
        "argmin_std": std_amin,
        "alpha_std": std_alpha,
        "beta_std": std_beta,
        "r_std": std_r,
    }

    return stats


cf_vol_axes_setting = {
    "title": "coarse-fine volatility",
    "ylabel": r"$\rho(k)$",
    "xlabel": "lag k",
    "xscale": "linear",
    "yscale": "linear",
}
cf_vol_plot_setting = {
    "alpha": 1,
    "marker": "o",
    "color": "royalblue",
    "markersize": 1,
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
