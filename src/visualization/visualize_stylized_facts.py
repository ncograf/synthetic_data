from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from coarse_fine_volatility import (
    cf_vol_axes_setting,
    cf_vol_plot_setting,
    corse_fine_volatility,
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


def visualize_stylized_facts(
    price_data: pd.DataFrame, return_data: pd.DataFrame = None
) -> plt.Figure:
    """Visualizes all stilized facts and returns the plt figure

    Args:
        price_data (pd.DataFrame):  n_timesteps x m_stocks
        return_data (pd.DataFrame, optional):  n_timesteps x m_stocks return data. Defaults to None

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
    if isinstance(price_data, pd.Series):
        price_data = price_data.to_frame()
    price_data = price_data.to_numpy()

    if return_data is None:
        log_returns = np.log(price_data[1:] / price_data[:-1])
    else:
        if isinstance(return_data, pd.Series):
            return_data = return_data.to_frame().to_numpy()
        log_returns = np.log(return_data)
    max_lag = log_returns.shape[0] - 1

    # LINEAR UNPREDICTABILITY
    linear_unpredict_data = linear_unpredictability(
        log_returns=log_returns, max_lag=min(1000, max_lag)
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
        log_returns=log_returns, max_lag=min(1000, max_lag)
    )
    axes[0, 2].set(**vol_clust_axes_setting)
    axes[0, 2].plot(np.mean(vol_clust_data, axis=1), **vol_clust_plot_setting)

    # LEVERAGE EFFECT
    leverage_effect_data = leverage_effect(log_returns=log_returns, max_lag=100)
    axes[1, 0].set(**lev_eff_axes_setting)
    axes[1, 0].plot(np.mean(leverage_effect_data, axis=1), **lev_eff_plot_setting)
    axes[1, 0].axhline(y=0, linestyle="--", c="black", alpha=0.4)

    # COARSE FINE VOLATILITY
    cf_vol_data, cf_vol_x, lead_lag_data, lead_lag_x = corse_fine_volatility(
        log_returns=log_returns, tau=5, max_lag=30
    )
    axes[1, 1].set(**cf_vol_axes_setting)
    axes[1, 1].plot(cf_vol_x, np.mean(cf_vol_data, axis=1), **cf_vol_plot_setting)
    axes[1, 1].plot(lead_lag_x, np.mean(lead_lag_data, axis=1), **lead_lag_plot_setting)
    axes[1, 1].legend(loc="lower left")
    axes[1, 1].axhline(y=0, linestyle="--", c="black", alpha=0.4)

    # GAIN LOSS ASYMMETRY
    gain_data, loss_data = gain_loss_asymmetry(
        price=price_data, max_lag=min(1000, max_lag), theta=0.1
    )
    axes[1, 2].set(**gain_loss_axis_setting)  # settings definitions are imported
    axes[1, 2].plot(np.mean(gain_data, axis=1), **gain_plot_setting)
    axes[1, 2].plot(np.mean(loss_data, axis=1), **loss_plot_setting)
    axes[1, 2].legend(loc="best")

    return fig


def visualize_averaged_stylized_facts(
    price_data_list: List[pd.DataFrame], return_data_list: List[pd.DataFrame] = None
) -> plt.Figure:
    """Visualizes all stilized facts and returns the plt figure averaged over the data in the list

    Note that alle element in the list must have the same

    Args:
        price_data (List[pd.DataFrame]):  List of prices n_timesteps x m_stocks
        return_data (List[pd.DataFrame], optional):  List of returns n_timesteps x m_stocks return data. Defaults to None

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
    for i, price_data in enumerate(price_data_list):
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame()
        price_data_list[i] = price_data.to_numpy()

    log_return_list = []
    if return_data_list is None:
        for price_data in price_data_list:
            log_return_list.append(np.log(price_data[:-1] / price_data[1:]))
    else:
        for return_data in return_data_list:
            if isinstance(return_data, pd.Series):
                return_data = return_data.to_frame().to_numpy()
            log_return_list.append(np.log(return_data))
    max_lag = log_return_list[0].shape[0] - 1

    # LINEAR UNPREDICTABILITY
    lin_unpred_list = []
    for log_returns in log_return_list:
        lin_unpred_list.append(
            linear_unpredictability(log_returns=log_returns, max_lag=min(1000, max_lag))
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
            volatility_clustering(log_returns=log_returns, max_lag=min(1000, max_lag))
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
    cv_vol_data_list, cv_vol_x_list, lead_lag_data_list, lead_lag_x_list = (
        [],
        [],
        [],
        [],
    )
    for log_returns in log_return_list:
        cv_dat, cv_x, ll_dat, ll_x = corse_fine_volatility(
            log_returns=log_returns, tau=5, max_lag=30
        )
        cv_vol_data_list.append(cv_dat)
        cv_vol_x_list.append(cv_x)
        lead_lag_data_list.append(ll_dat)
        lead_lag_x_list.append(ll_x)
    cf_vol_data = np.mean(cv_vol_data_list, axis=0)
    cf_vol_x = np.mean(cv_vol_x_list, axis=0)
    lead_lag_data = np.mean(lead_lag_data_list, axis=0)
    lead_lag_x = np.mean(lead_lag_x_list, axis=0)
    axes[1, 1].set(**cf_vol_axes_setting)
    axes[1, 1].plot(cf_vol_x, np.mean(cf_vol_data, axis=1), **cf_vol_plot_setting)
    axes[1, 1].plot(lead_lag_x, np.mean(lead_lag_data, axis=1), **lead_lag_plot_setting)
    axes[1, 1].legend(loc="lower left")
    axes[1, 1].axhline(y=0, linestyle="--", c="black", alpha=0.4)

    # GAIN LOSS ASYMMETRY
    gd_list, ld_list = [], []
    for price_data in price_data_list:
        gd, ld = gain_loss_asymmetry(
            price=price_data, max_lag=min(1000, max_lag), theta=0.1
        )
        gd_list.append(gd)
        ld_list.append(ld)
    gain_data, loss_data = np.mean(gd_list, axis=0), np.mean(ld_list, axis=0)
    axes[1, 2].set(**gain_loss_axis_setting)  # settings definitions are imported
    axes[1, 2].plot(np.mean(gain_data, axis=1), **gain_plot_setting)
    axes[1, 2].plot(np.mean(loss_data, axis=1), **loss_plot_setting)
    axes[1, 2].legend(loc="best")

    return fig
