import boosted_stats
import numpy as np
import pandas as pd


def corse_fine_volatility(log_returns: pd.DataFrame, tau: int, max_lag: int):
    log_returns = np.array(log_returns)

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
    "color": "blue",
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
