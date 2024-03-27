import click
import numpy as np
import data_inspector
import real_data_loader
from pathlib import Path
from typing import Optional, Set, List

import sp500_statistic
import base_statistic
import base_outlier_set
import log_return_statistic
import scaled_log_return_statistic
import wavelet_statistic
import stock_price_statistic
import spike_statistic
import isolation_forest_set
import outlier_summary
import cached_outlier_set
import garch_generator
import index_generator
import illiquidity_filter
import coarse_fine_volatility
import auto_corr_statistic
import leverage_effect
import statistic_inspector
import normalized_price_return
import abs_log_return_statistic
import gain_loss_asymetry
import return_statistic
import stock_price_statistic
import time


@click.group()
def inspect():
    pass

@inspect.command()
def visualize_stylized_fact():

    data_loader = real_data_loader.RealDataLoader()
    stock_data = data_loader.get_timeseries(col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False)
    n_data = stock_data.shape[0]
    # stock_data = stock_data.iloc[n_data - 10000:,:]
    
    start = time.time()
    log_stat = log_return_statistic.LogReturnStatistic(0.001)
    log_stat.set_statistics(stock_data)

    ret_stat = return_statistic.ReturnStatistic(0.001)
    ret_stat.set_statistics(stock_data)

    abs_log_stat = abs_log_return_statistic.AbsLogReturnStatistic(0.001)
    abs_log_stat.set_statistics(stock_data)

    price = stock_price_statistic.StockPriceStatistic(0.001)
    price.set_statistics(stock_data)

    m_lag = 1000
    log_corr_stat = auto_corr_statistic.AutoCorrStatistic(max_lag=m_lag, underlaying=log_stat, implementation='boosted')
    log_corr_stat.set_statistics(None)

    volatility_clustering = auto_corr_statistic.AutoCorrStatistic(max_lag=m_lag, underlaying=abs_log_stat, implementation='boosted')
    volatility_clustering.set_statistics(None)

    norm_price_ret = normalized_price_return.NormalizedPriceReturn(log_stat)
    norm_price_ret.set_statistics(None)
    
    lev_eff = leverage_effect.LeverageEffect(max_lag=100, underlaying=ret_stat)
    lev_eff.set_statistics(None)

    co_fine = coarse_fine_volatility.CoarseFineVolatility(max_lag=25, tau=5, underlaying=log_stat)
    co_fine.set_statistics(None)

    gain_loss_asym = gain_loss_asymetry.GainLossAsymetry(max_lag=m_lag, theta=0.1, underlying_price=price)
    gain_loss_asym.set_statistics(None)

    print(f"Time: {time.time() - start}") 

    inspector = statistic_inspector.StatisticInspector(cache="data/cache")
    figure_params = {"figure.figsize" : (16, 10),
             "font.size" : 24,
             "figure.dpi" : 96,
             "figure.constrained_layout.use" : True,
             "figure.constrained_layout.h_pad" : 0.1,
             "figure.constrained_layout.hspace" : 0,
             "figure.constrained_layout.w_pad" : 0.1,
             "figure.constrained_layout.wspace" : 0,
            }
    ax_params_auto_corr = {
        'yscale' : 'linear',
        'xscale' : 'log',
        'ylim' : (-1,1),
        'xlim' : (None, None),
        'xlabel' : 'lag k',
        'ylabel' : 'Auto-correlation',
    }
    inspector.plot_average_sylized_fact(
        stylized_fact=log_corr_stat, 
        rc_params=figure_params,
        ax_params=ax_params_auto_corr,
        )

    ax_params_volatility = {
        'yscale' : 'log',
        'xscale' : 'log',
        'ylim' : (None, None),
        'xlim' : (None, None),
        'xlabel' : 'lag k',
        'ylabel' : 'Auto-correlation',
    }
    inspector.plot_average_sylized_fact(
        stylized_fact=volatility_clustering, 
        rc_params=figure_params,
        ax_params=ax_params_volatility,
        )

    ax_params_heavy_tailed = {
        'yscale' : 'log',
        'xscale' : 'log',
        'ylim' : (None, None),
        'xlim' : (None, None),
        'ylabel' : r'$1 - F(r)$',
        'xlabel' : 'normalized price returns',
    }
    inspector.plot_average_sylized_fact(
        stylized_fact=norm_price_ret,
        rc_params=figure_params,
        ax_params=ax_params_heavy_tailed,
    )

    style_levarage = {
        'alpha' : 1,
        'marker' : 'o',
        'markersize' : 1,
        'linestyle' : '-',
        'color' : 'blue',
        'color_neg' : 'red',
        'color_pos' : 'blue',
    }
    ax_params_lev_eff = {
        'yscale' : 'linear',
        'xscale' : 'linear',
        'ylim' : (None, None),
        'xlim' : (None, None),
        'ylabel' : r'L(k)',
        'xlabel' : r'lag $k$',
    }
    inspector.plot_average_sylized_fact(
        stylized_fact=lev_eff,
        rc_params=figure_params,
        ax_params=ax_params_lev_eff,
        style=style_levarage,
    )

    ax_params_coarse_fine = {
        'yscale' : 'linear',
        'xscale' : 'linear',
        'ylim' : (None, None),
        'xlim' : (None, None),
        'ylabel' : r'$p(k)$',
        'xlabel' : r'lag $k$',
    }
    inspector.plot_average_sylized_fact(
        stylized_fact=co_fine,
        rc_params=figure_params,
        ax_params=ax_params_coarse_fine,
    )

    style_gain_loss = {
        'alpha' : 1,
        'marker' : 'o',
        'markersize' : 1,
        'linestyle' : 'None',
        'color' : 'blue',
        'color_neg' : 'red',
        'color_pos' : 'blue',
    }
    ax_params_gain_loss = {
        'yscale' : 'linear',
        'xscale' : 'log',
        'ylim' : (None, None),
        'xlim' : (None, None),
        'ylabel' : r'return time probability',
        'xlabel' : r"timestep t'",
    }
    inspector.plot_average_sylized_fact(
        stylized_fact=gain_loss_asym,
        rc_params=figure_params,
        ax_params=ax_params_gain_loss,
        style=style_gain_loss,
    )


if __name__ == "__main__":

    visualize_stylized_fact()