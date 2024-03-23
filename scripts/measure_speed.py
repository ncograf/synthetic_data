import click
import numpy as np
import data_inspector
import real_data_loader
from pathlib import Path
from typing import Optional, Set, List
import matplotlib.pyplot as plt

import sp500_statistic
import base_statistic
import base_outlier_set
import auto_corr_statistic
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
import time

@click.command()
def measure_abs_log_return():
    
    data_loader = real_data_loader.RealDataLoader()
    real_adj_close_data = data_loader.get_timeseries(col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False)
    log_stat = log_return_statistic.LogReturnStatistic(0.001)
    log_stat.set_statistics(real_adj_close_data)

    
    python_loop_times = []
    boosted_times = []
    python_strides_time = []
    max_lag = 1001
    lags = range(1, max_lag + 1)
    for i in lags:
        start = time.time()
        log_corr_stat = auto_corr_statistic.AutoCorrStatistic(max_lag=i, underlaying=log_stat, implementation='strides')
        log_corr_stat.set_statistics(None)
        stop = time.time()
        python_strides_time.append(stop - start)
        print(f"Time stride for {i} lags: {stop - start}") 

        start = time.time()
        log_corr_stat = auto_corr_statistic.AutoCorrStatistic(max_lag=i, underlaying=log_stat, implementation='boosted')
        log_corr_stat.set_statistics(None)
        stop = time.time()
        boosted_times.append(stop - start)
        print(f"Time boosted for {i} lags: {stop - start}") 

        start = time.time()
        log_corr_stat = auto_corr_statistic.AutoCorrStatistic(max_lag=i, underlaying=log_stat, implementation='python_loop')
        log_corr_stat.set_statistics(None)
        stop = time.time()
        python_loop_times.append(stop - start)
        print(f"Time python loop for {i} lags: {stop - start}") 
    
    figure_params = {"figure.figsize" : (16, 10),
             "font.size" : 24,
             "figure.dpi" : 96,
             "figure.constrained_layout.use" : True,
             "figure.constrained_layout.h_pad" : 0.1,
             "figure.constrained_layout.hspace" : 0,
             "figure.constrained_layout.w_pad" : 0.1,
             "figure.constrained_layout.wspace" : 0,
            }
    
    with plt.rc_context(rc=figure_params):
        plt.plot(lags, python_strides_time, label='Python stride implementation')
        plt.plot(lags, python_loop_times, label='Python loop implementation')
        plt.plot(lags, boosted_times, label='C++ boosted implementation')
        plt.legend()
        plt.savefig("data/cache/auto_correlation_speed.png")
        plt.show()
    
if __name__ == '__main__':
    measure_abs_log_return()