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
import historic_events

@click.group()
def inspect():
    pass

#TODO implement multicommand pipeline maybe https://click.palletsprojects.com/en/8.1.x/commands/#multi-command-pipelines

@click.command()
def outlier():
    
    data_loader = real_data_loader.RealDataLoader()
    real_adj_close_data = data_loader.get_timeseries(col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False)
    real_close_data = data_loader.get_timeseries(col_name="Close", data_path="data/raw_yahoo_data", update_all=False)
    real_open_data = data_loader.get_timeseries(col_name="Open", data_path="data/raw_yahoo_data", update_all=False)
    data = [real_adj_close_data, real_close_data, real_open_data]
    postfixes = [' - Adj Close', ' - Close', ' - Open']
    colors = ['green', 'blue', 'orange']
    
    statistics = []
    time_statistics= [] 
    outlier_detectors = set()

    quantile = 0.001
    

    log_stats = [log_return_statistic.LogReturnStatistic(quantile=quantile, legend_postfix=p, color=c) for p, c  in zip(postfixes, colors)]
    scaled_log_stats = [scaled_log_return_statistic.ScaledLogReturnStatistic(quantile=quantile, window=100, legend_postfix=p, color=c) for p, c  in zip(postfixes, colors)]
    price_stats = [stock_price_statistic.StockPriceStatistic(quantile=quantile, legend_postfix=p, color=c) for p, c in zip(postfixes, colors)]

    #spike_stat = spike_statistic.SpikeStatistic(denomiator_scaling=np.exp, quantile=quantile, function_name='exp')
    #forest_outlier = isolation_forest_set.IsolationForestStatisticSet(quantile=quantile, statistic=log_stat)

    statistics.append(log_stats[0])
    statistics.append(scaled_log_stats[0])
    #statistics.append(spike_stat)
    #statistics.append(wave_stat_3)
    
    time_statistics.append(log_stats)
    time_statistics.append(price_stats)
    time_statistics.append(scaled_log_stats)

    #outlier_detectors.add(cached_det)
    #outlier_detectors.add(wave_stat_3)
    outlier_detectors.add(log_stats[0])
    outlier_detectors.add(log_stats[1])
    outlier_detectors.add(scaled_log_stats[0])
    outlier_detectors.add(scaled_log_stats[1])
    #outlier_detectors.add(spike_stat)
    #outlier_detectors.add(forest_outlier)
        
    for stats in time_statistics:
        for stat, dat in zip(stats, data):
            stat.set_statistics(dat)

    for stat in statistics:
        stat.set_statistics(real_adj_close_data)
    
    # some outlier might depend on the statistic so this needs to be after set statstics
    # TODO for pure outlier statistics make sure to set the data

    historic_events_path = f"data/cache/historical_events.json"
    outlier_path = f"data/cache/outlier_json.json"
    
    inspector = data_inspector.DataInspector()
    inspector.check_outliers_manually(statistics=statistics,
                                    time_statistics=time_statistics,
                                    outlier_detectors=outlier_detectors,
                                    outlier_path=outlier_path,
                                    historic_events_path=historic_events_path,
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
    inspector.plot_histogram(data_statistic, rc_params=figure_params, density=True, copy=copy)


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

@click.command()
def visualize_garch_data():

    data_loader = real_data_loader.RealDataLoader()
    real_stock_data = data_loader.get_timeseries(col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False)
    real_stock_data = real_stock_data.loc[:,real_stock_data.columns[:20]]

    garch = garch_generator.GarchGenerator()
    index_gen = index_generator.IndexGenerator(generator=garch)
    synth_data = index_gen.generate_index(real_stock_data)
    
    real_data = stock_price_statistic.StockPriceStatistic(quantile=0.1)
    real_data.set_statistics(real_stock_data)
    real_log = log_return_statistic.LogReturnStatistic(quantile=0.1)
    real_log.set_statistics(real_stock_data)

    synthetic_data = stock_price_statistic.StockPriceStatistic(quantile=0.1, legend_postfix=" - GARCH")
    synthetic_data.set_statistics(synth_data)
    synthetic_log = log_return_statistic.LogReturnStatistic(quantile=0.1, legend_postfix=" - GARCH")
    synthetic_log.set_statistics(synth_data)

    figure_params = {"figure.figsize" : (16, 10),
             "font.size" : 24,
             "figure.dpi" : 96,
             "figure.constrained_layout.use" : True,
             "figure.constrained_layout.h_pad" : 0.1,
             "figure.constrained_layout.hspace" : 0,
             "figure.constrained_layout.w_pad" : 0.1,
             "figure.constrained_layout.wspace" : 0,
            }

    inspector = data_inspector.DataInspector()
    statistics = [[synthetic_data, real_data],[synthetic_log, real_log]]

    for symbol in real_stock_data.columns:
        inspector.plot_time_series(statistics=statistics, symbols=[symbol], rc_params=figure_params)


if __name__ == "__main__":
    #sumarize_outliers()

    outlier()
    #visualize_garch_data()

    #runner = CliRunner()
    #result = runner.invoke(inspect_raw_data, ["--copy", "/home/nico/thesis/protocol/figures"])