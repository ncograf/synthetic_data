import click
import numpy as np
import data_inspector
import real_data_loader
from pathlib import Path
from typing import Optional
from click.testing import CliRunner

import sp500_statistic
import log_return_statistic
import wavelet_statistic
import stock_price_statistic
import spike_statistic
import isolation_forest_set
import outlier_summary
import cached_outlier_set

@click.group()
def inspect():
    pass

#TODO implement multicommand pipeline maybe https://click.palletsprojects.com/en/8.1.x/commands/#multi-command-pipelines

@click.command()
def outlier():
    
    data_loader = real_data_loader.RealDataLoader()
    real_stock_data = data_loader.get_timeseries(col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False)
    
    statistics = set()
    time_statistics = set()
    outlier_detectors = set()

    quantile = 0.001

    log_stat = log_return_statistic.LogReturnStatistic(quantile=quantile)
    price_stat = stock_price_statistic.StockPriceStatistic(quantile=quantile)
    spike_stat = spike_statistic.SpikeStatistic(denomiator_scaling=np.exp, quantile=quantile, function_name='exp')
    
    forest_outlier = isolation_forest_set.IsolationForestStatisticSet(quantile=quantile, statistic=log_stat)

    pattern = np.array([-1,2,-1])
    wavelet_simple = wavelet_statistic.Wavelet(pattern)
    wave_stat_3 = wavelet_statistic.WaveletStatistic(wavelet=wavelet_simple, normalize=False, quantile=quantile)

    n = 10
    pattern = np.zeros(n + 1)
    pattern[:n//2] = -np.arange(n//2)
    pattern[n//2+1:] = np.flip(-np.arange(n//2))
    weight = pattern.sum()
    pattern[n//2] = -weight
    pattern = pattern / weight
    wavelet_wide = wavelet_statistic.Wavelet(pattern)
    wave_stat_n = wavelet_statistic.WaveletStatistic(wavelet=wavelet_wide, normalize=True, quantile=quantile)
    cached_det = cached_outlier_set.CachedOutlierSet("data/cache/presentation_outlier.json")
    
    statistics.add(log_stat)
    #statistics.add(spike_stat)
    #statistics.add(wave_stat_3)
    
    time_statistics.add(log_stat)
    time_statistics.add(price_stat)
    #time_statistics.add(spike_stat)

    #outlier_detectors.add(cached_det)
    #outlier_detectors.add(wave_stat_3)
    outlier_detectors.add(log_stat)
    #outlier_detectors.add(spike_stat)
    #outlier_detectors.add(forest_outlier)
    
    inspector = data_inspector.DataInspector(data=real_stock_data)
    inspector.check_outliers_manually(statistics=statistics,
                                    time_statistics=time_statistics,
                                    outlier_detectors=outlier_detectors,
                                    outlier_path="data/cache/precentile_outlier.json",
                                    n_context=0,
                                    )

@click.command()
@click.option('-c','--copy', type=click.Path(exists=True,file_okay=False,dir_okay=True,writable=True), required=False)
def inspect_raw_data(copy : Optional[Path] = None):
    figure_params = {"figure.figsize" : (16, 10),
             "font.size" : 24,
             "figure.dpi" : 96,
             "figure.constrained_layout.use" : True,
             "figure.constrained_layout.h_pad" : 0.1,
             "figure.constrained_layout.hspace" : 0,
             "figure.constrained_layout.w_pad" : 0.1,
             "figure.constrained_layout.wspace" : 0,
            }
    data_loader = real_data_loader.RealDataLoader()
    real_stock_data = data_loader.get_timeseries(col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False)
    data_statistic = sp500_statistic.SP500Statistic()
    data_statistic.set_statistics(real_stock_data)
    data_statistic.print_distribution_properties() # TODO move the function to the inspector
    inspector = data_inspector.DataInspector(data=real_stock_data)
    inspector.plot_histogram(data_statistic, rc_params=figure_params, density=False, copy=copy)


@click.command()
def sumarize_outliers():

    data_loader = real_data_loader.RealDataLoader()
    real_stock_data = data_loader.get_timeseries(col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False)

    quantile = 0.001
    log_stat = log_return_statistic.LogReturnStatistic(quantile=quantile)
    spike_stat = spike_statistic.SpikeStatistic(denomiator_scaling=np.exp, quantile=quantile, function_name='exp')
    cache_det = cached_outlier_set.CachedOutlierSet(path="data/cache/precentile_outlier.json")
    


    summary = outlier_summary.OutlierSummary(data=real_stock_data, detectors=[cache_det])
    summary.print_outlier_distribution()
    

if __name__ == "__main__":
    #sumarize_outliers()

    outlier()

    #runner = CliRunner()
    #result = runner.invoke(inspect_raw_data, ["--copy", "/home/nico/thesis/protocol/figures"])