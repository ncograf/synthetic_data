from typing import Any, Dict, Tuple

import boosted_stats
import lagged_correlation
import numpy as np
import numpy.typing as npt
import torch
from power_fit import fit_powerlaw


def coarse_fine_volatility(
    log_returns: torch.Tensor, tau: int, max_lag: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    numpy = False
    if not torch.is_tensor(log_returns):
        numpy = True
        log_returns = torch.tensor(log_returns)

    nan_mask_shifted = torch.isnan(log_returns[:-tau])

    # compute coarse volatiltiy |sum r_t|
    cum_sum = torch.cumsum(log_returns, dim=0)
    v_c_tau = torch.abs(cum_sum[tau:] - cum_sum[:-tau])
    v_c_tau[nan_mask_shifted] = torch.nan

    # compute fine volatiltiy sum |r_t|
    abs_log_return = torch.abs(log_returns)
    abs_cumsum = torch.cumsum(abs_log_return, dim=0)
    v_f_tau = abs_cumsum[tau:] - abs_cumsum[:-tau]
    v_f_tau = v_f_tau / tau
    v_f_tau[nan_mask_shifted] = torch.nan

    stat_pos = lagged_correlation.lagged_corr(v_f_tau, v_c_tau, max_lag, dim=0)
    stat_neg = lagged_correlation.lagged_corr(v_c_tau, v_f_tau, max_lag, dim=0)

    # normalize to get correlation
    v_f_std = torch.sqrt(
        torch.nanmean((v_f_tau - torch.nanmean(v_f_tau, dim=0)) ** 2, dim=0)
    )
    v_c_std = torch.sqrt(
        torch.nanmean((v_c_tau - torch.nanmean(v_c_tau, dim=0)) ** 2, dim=0)
    )
    stat_neg = stat_neg / (v_c_std * v_f_std)
    stat_pos = stat_pos / (v_c_std * v_f_std)

    stat = torch.cat([torch.flip(stat_neg, dims=[0]), stat_pos[1:]], dim=0)
    lead_lag = stat_pos - stat_neg

    lead_lag_x = torch.arange(max_lag + 1)
    stat_x = torch.concatenate([-torch.flip(lead_lag_x, dims=[0]), lead_lag_x[1:]])

    if numpy:
        return (
            np.asarray(stat),
            np.asarray(stat_x),
            np.asarray(lead_lag),
            np.asarray(lead_lag_x),
        )

    return stat, stat_x, lead_lag, lead_lag_x


def coarse_fine_volatility_depr(
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
    delta_lead_lag = stat_pos - stat_neg

    delta_lead_lag_x = np.arange(max_lag + 1)
    stat_x = np.concatenate([-np.flip(delta_lead_lag_x), delta_lead_lag_x[1:]])

    return stat, stat_x, delta_lead_lag, delta_lead_lag_x


def coarse_fine_volatility_stats(
    log_returns: npt.ArrayLike, tau: int, max_lag: int
) -> Dict[str, Any]:
    """Coarse Fine Volatility Statistics

    :math:
        p(k) = Corr(v_c(t+k), v_f)
        v_c(t) = | \sum_i r_{t - i} |
        v_f(t) = \sum_i | r_{t - i} |

        \Delta p(k) = p(k) - p(-k)

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
