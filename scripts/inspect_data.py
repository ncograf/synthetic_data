from pathlib import Path
from typing import Optional

import cached_outlier_set
import click
import data_inspector
import garch_generator
import gen_data_loader
import illiquidity_filter
import log_return_statistic
import numpy as np
import outlier_summary
import real_data_loader
import sp500_statistic
import stock_price_statistic


@click.group()
def inspect():
    pass


# TODO implement multicommand pipeline maybe https://click.palletsprojects.com/en/8.1.x/commands/#multi-command-pipelines


@click.command()
def outlier():
    data_loader = real_data_loader.RealDataLoader()
    real_adj_close_data = data_loader.get_timeseries(
        col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False
    )
    real_close_data = data_loader.get_timeseries(
        col_name="Close", data_path="data/raw_yahoo_data", update_all=False
    )
    real_open_data = data_loader.get_timeseries(
        col_name="Open", data_path="data/raw_yahoo_data", update_all=False
    )
    data = [real_adj_close_data, real_close_data, real_open_data]
    postfixes = [" - Adj Close", " - Close", " - Open"]
    colors = ["green", "blue", "orange"]

    statistics = []
    time_statistics = []
    outlier_detectors = set()

    quantile = 0.001

    log_stats = [
        log_return_statistic.LogReturnStatistic(
            quantile=quantile, legend_postfix=p, color=c
        )
        for p, c in zip(postfixes, colors)
    ]
    # scaled_log_stats = [scaled_log_return_statistic.ScaledLogReturnStatistic(quantile=quantile, window=100, legend_postfix=p, color=c) for p, c  in zip(postfixes, colors)]
    price_stats = [
        stock_price_statistic.StockPriceStatistic(
            quantile=quantile, legend_postfix=p, color=c
        )
        for p, c in zip(postfixes, colors)
    ]

    # spike_stat = spike_statistic.SpikeStatistic(denomiator_scaling=np.exp, quantile=quantile, function_name='exp')
    # forest_outlier = isolation_forest_set.IsolationForestStatisticSet(quantile=quantile, statistic=log_stat)

    statistics.append(log_stats[0])
    # statistics.append(scaled_log_stats[0])
    # statistics.append(spike_stat)
    # statistics.append(wave_stat_3)

    time_statistics.append(log_stats)
    time_statistics.append(price_stats)
    # time_statistics.append(scaled_log_stats)

    # outlier_detectors.add(cached_det)
    # outlier_detectors.add(wave_stat_3)
    outlier_detectors.add(log_stats[0])
    outlier_detectors.add(log_stats[1])
    # outlier_detectors.add(scaled_log_stats[0])
    # outlier_detectors.add(scaled_log_stats[1])
    # outlier_detectors.add(spike_stat)
    # outlier_detectors.add(forest_outlier)

    for stats in time_statistics:
        for stat, dat in zip(stats, data):
            stat.set_statistics(dat)

    for stat in statistics:
        stat.set_statistics(real_adj_close_data)

    # some outlier might depend on the statistic so this needs to be after set statstics
    # TODO for pure outlier statistics make sure to set the data

    historic_events_path = "data/cache/historical_events.json"
    outlier_path = "data/cache/outlier_json.json"

    inspector = data_inspector.DataInspector()
    inspector.check_outliers_manually(
        statistics=statistics,
        time_statistics=time_statistics,
        outlier_detectors=outlier_detectors,
        outlier_path=outlier_path,
        historic_events_path=historic_events_path,
        n_context=0,
    )


@click.command()
@click.option(
    "-c",
    "--copy",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    required=False,
)
def inspect_raw_data(copy: Optional[Path] = None):
    figure_params = {
        "figure.figsize": (16, 10),
        "font.size": 24,
        "figure.dpi": 96,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.1,
        "figure.constrained_layout.hspace": 0,
        "figure.constrained_layout.w_pad": 0.1,
        "figure.constrained_layout.wspace": 0,
    }
    data_loader = real_data_loader.RealDataLoader()
    real_stock_data = data_loader.get_timeseries(
        col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False
    )
    data_statistic = sp500_statistic.SP500Statistic()
    data_statistic.set_statistics(real_stock_data)
    data_statistic.print_distribution_properties()  # TODO move the function to the inspector
    inspector = data_inspector.DataInspector(data=real_stock_data)
    inspector.plot_histogram(
        data_statistic, rc_params=figure_params, density=True, copy=copy
    )


@click.command()
@click.option(
    "-c",
    "--copy",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    required=False,
)
def summarize_liquid_data(copy: Optional[Path] = None):
    figure_params = {
        "figure.figsize": (16, 10),
        "font.size": 24,
        "figure.dpi": 96,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.1,
        "figure.constrained_layout.hspace": 0,
        "figure.constrained_layout.w_pad": 0.1,
        "figure.constrained_layout.wspace": 0,
    }
    data_loader = real_data_loader.RealDataLoader()
    real_stock_data = data_loader.get_timeseries(
        col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False
    )
    filter = illiquidity_filter.IlliquidityFilter(window=5, min_jumps=1)
    liquid_data = filter.get_data(real_stock_data)
    data_statistic = sp500_statistic.SP500Statistic()
    data_statistic.set_statistics(liquid_data)
    data_statistic.print_distribution_properties()  # TODO move the function to the inspector
    inspector = data_inspector.DataInspector()
    inspector.plot_histogram(
        data_statistic,
        rc_params=figure_params,
        symbol="num_data",
        density=False,
        copy=copy,
    )


@click.command()
def sumarize_outliers():
    data_loader = real_data_loader.RealDataLoader()
    real_stock_data = data_loader.get_timeseries(
        col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False
    )

    cache_det = cached_outlier_set.CachedOutlierSet(
        path="data/cache/precentile_outlier.json"
    )

    summary = outlier_summary.OutlierSummary(
        data=real_stock_data, detectors=[cache_det]
    )
    summary.print_outlier_distribution()


def visualize_data(symbol):
    data_loader = real_data_loader.RealDataLoader()
    real_stock_data = data_loader.get_timeseries(
        col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False
    )

    real_data = stock_price_statistic.StockPriceStatistic(quantile=0.1)
    real_data.set_statistics(real_stock_data)
    figure_params = {
        "figure.figsize": (16, 10),
        "font.size": 24,
        "figure.dpi": 96,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.1,
        "figure.constrained_layout.hspace": 0,
        "figure.constrained_layout.w_pad": 0.1,
        "figure.constrained_layout.wspace": 0,
    }

    inspector = data_inspector.DataInspector()
    inspector.plot_time_series(
        statistics=[{real_data}], symbols=[symbol], rc_params=figure_params
    )


def visualize_garch_data(symbol):
    data_loader = real_data_loader.RealDataLoader()
    real_stock_data = data_loader.get_timeseries(
        col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False
    )
    # real_stock_data = real_stock_data.loc[:,real_stock_data.columns[:20]]

    garch = garch_generator.GarchGenerator(name="GARCH_1_1_normal")
    gen_loader = gen_data_loader.GenDataLoader()
    synth_data = gen_loader.get_timeseries(
        garch, data_loader=data_loader, col_name="Adj Close"
    )

    real_data = stock_price_statistic.StockPriceStatistic(quantile=0.1)
    real_data.set_statistics(real_stock_data)
    real_log = log_return_statistic.LogReturnStatistic(quantile=0.1)
    real_log.set_statistics(real_stock_data)

    synthetic_data = stock_price_statistic.StockPriceStatistic(
        quantile=0.1, legend_postfix=" - GARCH"
    )
    synthetic_data.set_statistics(synth_data)
    synthetic_log = log_return_statistic.LogReturnStatistic(
        quantile=0.1, legend_postfix=" - GARCH"
    )
    synthetic_log.set_statistics(synth_data)

    # Hack data
    stat = synthetic_log._statistic
    stat[np.isinf(stat)] = 0
    stat = synthetic_data._statistic
    stat[np.isinf(stat)] = 0

    figure_params = {
        "figure.figsize": (16, 10),
        "font.size": 24,
        "figure.dpi": 96,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.1,
        "figure.constrained_layout.hspace": 0,
        "figure.constrained_layout.w_pad": 0.1,
        "figure.constrained_layout.wspace": 0,
    }

    inspector = data_inspector.DataInspector()
    statistics = [[synthetic_data, real_data], [synthetic_log, real_log]]

    inspector.plot_time_series(
        statistics=statistics, symbols=[symbol], rc_params=figure_params
    )


if __name__ == "__main__":
    # sumarize_outliers()

    # summarize_liquid_data()
    visualize_data("ORCL")
    visualize_data("ODFL")
    visualize_data("MTCH")
    visualize_data("CTSH")
    visualize_data("C")
    visualize_garch_data("ORCL")
    visualize_garch_data("ODFL")
    visualize_garch_data("MTCH")
    visualize_garch_data("CTSH")
    visualize_garch_data("C")
    # ['C', 'CTSH', 'MTCH', 'ODFL', 'ORCL']

    # outlier()
    # visualize_garch_data()

    # runner = CliRunner()
    # result = runner.invoke(inspect_raw_data, ["--copy", "/home/nico/thesis/protocol/figures"])
