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

    log_stat = log_return_statistic.LogReturnStatistic()
    price_stat = stock_price_statistic.StockPriceStatistic()

    pattern = np.array([-1,2,-1])
    wavelet_simple = wavelet_statistic.Wavelet(pattern)
    wave_stat_3 = wavelet_statistic.WaveletStatistic(wavelet=wavelet_simple, normalize=False)

    n = 10
    pattern = np.zeros(n + 1)
    pattern[:n//2] = -np.arange(n//2)
    pattern[n//2+1:] = np.flip(-np.arange(n//2))
    weight = pattern.sum()
    pattern[n//2] = -weight
    pattern = pattern / weight
    wavelet_wide = wavelet_statistic.Wavelet(pattern)
    wave_stat_n = wavelet_statistic.WaveletStatistic(wavelet=wavelet_wide, normalize=True)
    
    statistics.add(log_stat)
    statistics.add(wave_stat_3)
    statistics.add(wave_stat_n)
    
    time_statistics.add(log_stat)
    time_statistics.add(price_stat)

    outlier_detectors.add(log_stat)
    outlier_detectors.add(wave_stat_3)
    outlier_detectors.add(wave_stat_n)
    
    inspector = data_inspector.DataInspector(data=real_stock_data)
    inspector.check_outliers_manually(statistics=statistics,
                                    time_statistics=time_statistics,
                                    outlier_detectors=outlier_detectors,
                                    outlier_path="data/cache/precentile_outlier.json",
                                    n_context=0,
                                    quantile=0.001)

@click.command()
@click.option('-c','--copy', type=click.Path(exists=True,file_okay=False,dir_okay=True,writable=True), required=False)
def raw_data(copy : Optional[Path] = None):
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
    

if __name__ == "__main__":
    outlier()
    #runner = CliRunner()
    #result = runner.invoke(raw_data, ["--copy", "/home/nico/thesis/protocol/figures"])