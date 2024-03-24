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
import auto_corr_statistic
import statistic_inspector
import normalized_price_return
import abs_log_return_statistic
import time


@click.group()
def inspect():
    pass

@inspect.command()
def visualize_stylized_fact():

    data_loader = real_data_loader.RealDataLoader()
    stock_data = data_loader.get_timeseries(col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False)
    
    start = time.time()
    log_stat = log_return_statistic.LogReturnStatistic(0.001)
    log_stat.set_statistics(stock_data)

    log_corr_stat = auto_corr_statistic.AutoCorrStatistic(max_lag=10, underlaying=log_stat, implementation='boosted')
    log_corr_stat.set_statistics(None)

    norm_price_ret = normalized_price_return.NormalizedPriceReturn(log_stat)
    norm_price_ret.set_statistics(None)

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


if __name__ == "__main__":

    visualize_stylized_fact()