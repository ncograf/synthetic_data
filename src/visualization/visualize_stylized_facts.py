from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from coarse_fine_volatility import (
    cf_vol_axes_setting,
    cf_vol_plot_setting,
    coarse_fine_volatility,
    lead_lag_plot_setting,
)
from gain_loss_asymetry import (
    gain_loss_asymmetry,
    gain_loss_axis_setting,
    gain_plot_setting,
    loss_plot_setting,
)
from heavy_tails import (
    heavy_tail_axes_setting,
    heavy_tail_neg_plot_setting,
    heavy_tail_pos_plot_setting,
    heavy_tails,
)
from leverage_effect import lev_eff_axes_setting, lev_eff_plot_setting, leverage_effect
from linear_unpredictability import (
    lin_unpred_plot_setting,
    lin_upred_axes_setting,
    linear_unpredictability,
)
from volatility_clustering import (
    vol_clust_axes_setting,
    vol_clust_plot_setting,
    volatility_clustering,
)


def visualize_stylized_facts(log_returns: npt.ArrayLike) -> plt.Figure:
    """Visualizes all stilized facts and returns the plt figure

    Args:
        log_returns (array_like):  (n_timesteps x m_stocks) or (n_timesteps) return data.

    Returns:
        plt.Figure: matplotlib figure ready to plot / save
    """

    # configure plt plots
    figure_style = {
        "text.usetex": True,
        "figure.figsize": (16, 10),
        "figure.titlesize": 22,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "font.size": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "figure.dpi": 96,
        "legend.loc": "upper right",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.1,
        "figure.constrained_layout.hspace": 0,
        "figure.constrained_layout.w_pad": 0.1,
        "figure.constrained_layout.wspace": 0,
    }
    subplot_layout = {
        "ncols": 3,
        "nrows": 2,
        "sharex": "none",
        "sharey": "none",
    }

    plt.rcParams.update(figure_style)
    fig, axes = plt.subplots(**subplot_layout, constrained_layout=True)

    # prepare data
    log_returns = np.array(log_returns)
    if log_returns.ndim == 1:
        log_returns = log_returns.reshape((-1, 1))
    elif log_returns.ndim > 2:
        raise RuntimeError(f"Log Returns has {log_returns.ndim} dimensions.")

    log_price = np.cumsum(log_returns, axis=0)
    max_ticks = log_returns.shape[0] - 1

    # LINEAR UNPREDICTABILITY
    linear_unpredict_data = linear_unpredictability(
        log_returns=log_returns, max_lag=min(1000, max_ticks)
    )
    axes[0, 0].set(**lin_upred_axes_setting)
    axes[0, 0].plot(np.mean(linear_unpredict_data, axis=1), **lin_unpred_plot_setting)

    # HEAVY TAILS
    (
        heavy_tail_density_pos,
        heavy_tail_bins_pos,
        heavy_tail_density_neg,
        heavy_tail_bins_neg,
    ) = heavy_tails(log_returns=log_returns)
    axes[0, 1].set(**heavy_tail_axes_setting)
    axes[0, 1].plot(
        heavy_tail_bins_pos, heavy_tail_density_pos, **heavy_tail_pos_plot_setting
    )
    axes[0, 1].plot(
        heavy_tail_bins_neg, heavy_tail_density_neg, **heavy_tail_neg_plot_setting
    )
    axes[0, 1].legend(loc="best")

    # VOLATILTIY CLUSTERING
    vol_clust_data = volatility_clustering(
        log_returns=log_returns, max_lag=min(1000, max_ticks)
    )
    axes[0, 2].set(**vol_clust_axes_setting)
    axes[0, 2].plot(np.mean(vol_clust_data, axis=1), **vol_clust_plot_setting)

    # LEVERAGE EFFECT
    leverage_effect_data = leverage_effect(log_returns=log_returns, max_lag=100)
    axes[1, 0].set(**lev_eff_axes_setting)
    axes[1, 0].plot(np.mean(leverage_effect_data, axis=1), **lev_eff_plot_setting)
    axes[1, 0].axhline(y=0, linestyle="--", c="black", alpha=0.4)

    # COARSE FINE VOLATILITY
    lead_lag_data, lead_lag_x, delta_lead_lag_data, delta_lead_lag_x = (
        coarse_fine_volatility(log_returns=log_returns, tau=5, max_lag=30)
    )
    axes[1, 1].set(**cf_vol_axes_setting)
    axes[1, 1].plot(lead_lag_x, np.mean(lead_lag_data, axis=1), **cf_vol_plot_setting)
    axes[1, 1].plot(
        delta_lead_lag_x, np.mean(delta_lead_lag_data, axis=1), **lead_lag_plot_setting
    )
    axes[1, 1].legend(loc="lower left")
    axes[1, 1].axhline(y=0, linestyle="--", c="black", alpha=0.4)

    # GAIN LOSS ASYMMETRY
    gain_data, loss_data = gain_loss_asymmetry(
        log_price=log_price, max_lag=min(1000, max_ticks), theta=0.1
    )
    axes[1, 2].set(**gain_loss_axis_setting)  # settings definitions are imported
    axes[1, 2].plot(np.mean(gain_data, axis=1), **gain_plot_setting)
    axes[1, 2].plot(np.mean(loss_data, axis=1), **loss_plot_setting)
    axes[1, 2].legend(loc="best")

    return fig


def visualize_averaged_stylized_facts(
    log_return_list: List[npt.ArrayLike],
) -> plt.Figure:
    """Visualizes all stilized facts and returns the plt figure averaged over the data in the list

    Note that alle element in the list must have the same

    Args:
        log_return_list (List[array_like]):  List of log returns (n_timesteps x m_stocks) or (n_timesteps) return data.

    Returns:
        plt.Figure: matplotlib figure ready to plot / save
    """

    # configure plt plots
    figure_style = {
        "text.usetex": True,
        "figure.figsize": (16, 10),
        "figure.titlesize": 22,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "font.size": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "figure.dpi": 96,
        "legend.loc": "upper right",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.1,
        "figure.constrained_layout.hspace": 0,
        "figure.constrained_layout.w_pad": 0.1,
        "figure.constrained_layout.wspace": 0,
    }
    subplot_layout = {
        "ncols": 3,
        "nrows": 2,
        "sharex": "none",
        "sharey": "none",
    }

    plt.rcParams.update(figure_style)
    fig, axes = plt.subplots(**subplot_layout, constrained_layout=True)

    log_price_list = []
    for idx, log_returns in enumerate(log_return_list):
        log_returns = np.array(log_returns)
        if log_returns.ndim == 1:
            log_returns = log_returns.reshape((-1, 1))
        elif log_returns.ndim > 2:
            raise RuntimeError(
                f"Log Returns in list index {idx} has {log_returns.ndim} dimensions."
            )

        log_price_list.append(np.cumsum(log_returns, axis=0))

    max_ticks = log_return_list[0].shape[0] - 1

    # LINEAR UNPREDICTABILITY
    lin_unpred_list = []
    for log_returns in log_return_list:
        lin_unpred_list.append(
            linear_unpredictability(
                log_returns=log_returns, max_lag=min(1000, max_ticks)
            )
        )
    lin_unpred_data = np.mean(lin_unpred_list, axis=0)
    axes[0, 0].set(**lin_upred_axes_setting)
    axes[0, 0].plot(np.mean(lin_unpred_data, axis=1), **lin_unpred_plot_setting)

    # HEAVY TAILS
    ht_dens_pos_list, ht_bin_pos_list = [], []
    ht_dens_neg_list, ht_bin_neg_list = [], []
    for log_returns in log_return_list:
        pos_dens, pos_bin, neg_dens, neg_bin = heavy_tails(log_returns=log_returns)
        ht_dens_pos_list.append(pos_dens)
        ht_bin_pos_list.append(pos_bin)
        ht_dens_neg_list.append(neg_dens)
        ht_bin_neg_list.append(neg_bin)
    (
        heavy_tail_bins_pos,
        heavy_tail_density_pos,
        heavy_tail_bins_neg,
        heavy_tail_density_neg,
    ) = (
        np.mean(ht_bin_pos_list, axis=0),
        np.mean(ht_dens_pos_list, axis=0),
        np.mean(ht_bin_neg_list, axis=0),
        np.mean(ht_dens_neg_list, axis=0),
    )
    axes[0, 1].set(**heavy_tail_axes_setting)
    axes[0, 1].plot(
        heavy_tail_bins_pos, heavy_tail_density_pos, **heavy_tail_pos_plot_setting
    )
    axes[0, 1].plot(
        heavy_tail_bins_neg, heavy_tail_density_neg, **heavy_tail_neg_plot_setting
    )
    axes[0, 1].legend(loc="best")

    # VOLATILTIY CLUSTERING
    vol_clust_list = []
    for log_returns in log_return_list:
        vol_clust_list.append(
            volatility_clustering(log_returns=log_returns, max_lag=min(1000, max_ticks))
        )
    vol_clust_data = np.mean(vol_clust_list, axis=0)
    axes[0, 2].set(**vol_clust_axes_setting)
    axes[0, 2].plot(np.mean(vol_clust_data, axis=1), **vol_clust_plot_setting)

    # LEVERAGE EFFECT
    lev_eff_list = []
    for log_returns in log_return_list:
        lev_eff_list.append(leverage_effect(log_returns=log_returns, max_lag=100))
    leverage_effect_data = np.mean(lev_eff_list, axis=0)
    axes[1, 0].set(**lev_eff_axes_setting)
    axes[1, 0].plot(np.mean(leverage_effect_data, axis=1), **lev_eff_plot_setting)
    axes[1, 0].axhline(y=0, linestyle="--", c="black", alpha=0.4)

    # COARSE FINE VOLATILITY
    ll_data_list, ll_x_list, delta_ll_data_list, delta_ll_x_list = (
        [],
        [],
        [],
        [],
    )
    for log_returns in log_return_list:
        ll_data, ll_x, delta_ll_data, delta_ll_x = coarse_fine_volatility(
            log_returns=log_returns, tau=5, max_lag=30
        )
        ll_data_list.append(ll_data)
        ll_x_list.append(ll_x)
        delta_ll_data_list.append(delta_ll_data)
        delta_ll_x_list.append(delta_ll_x)
    ll_data = np.mean(ll_data_list, axis=0)
    ll_x = np.mean(ll_x_list, axis=0)
    delta_ll_data = np.mean(delta_ll_data_list, axis=0)
    delta_ll_x = np.mean(delta_ll_x_list, axis=0)
    axes[1, 1].set(**cf_vol_axes_setting)
    axes[1, 1].plot(ll_x, np.mean(ll_data, axis=1), **cf_vol_plot_setting)
    axes[1, 1].plot(delta_ll_x, np.mean(delta_ll_data, axis=1), **lead_lag_plot_setting)
    axes[1, 1].legend(loc="lower left")
    axes[1, 1].axhline(y=0, linestyle="--", c="black", alpha=0.4)

    # GAIN LOSS ASYMMETRY
    gd_list, ld_list = [], []
    for log_prices in log_price_list:
        gd, ld = gain_loss_asymmetry(
            log_price=log_prices, max_lag=min(1000, max_ticks), theta=0.1
        )
        gd_list.append(gd)
        ld_list.append(ld)
    gain_data, loss_data = np.mean(gd_list, axis=0), np.mean(ld_list, axis=0)
    axes[1, 2].set(**gain_loss_axis_setting)  # settings definitions are imported
    axes[1, 2].plot(np.mean(gain_data, axis=1), **gain_plot_setting)
    axes[1, 2].plot(np.mean(loss_data, axis=1), **loss_plot_setting)
    axes[1, 2].legend(loc="best")

    return fig
