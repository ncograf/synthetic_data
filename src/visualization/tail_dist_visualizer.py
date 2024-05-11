import distribution_comparator
import pandas as pd
import plotter
import stock_price_statistic
import tail_statistics


def visualize_tail(
    time_series: pd.DataFrame,
    real_series: pd.DataFrame,
    plot_name: str,
    quantile: float = 0.01,
) -> plotter.Plotter:
    """Plot a stylized fact for two time series data sets, note that
    this implies that for example log returns have to be computed beforehand.

    Args:
        real_data (pd.DataFrame): First data set refered to as real
        gen_data (pd.DataFrame): Second data set refered to as fake
        fact (Literal): Stylized fact.

    Returns:
        Plotter: Plotter containing the plot.
    """

    plot = plotter.Plotter(
        cache="data/cache",
        figure_style={
            "figure.figsize": (16, 10),
            "figure.titlesize": 24,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "font.size": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "figure.dpi": 96,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.1,
            "figure.constrained_layout.hspace": 0,
            "figure.constrained_layout.w_pad": 0.1,
            "figure.constrained_layout.wspace": 0,
        },
        figure_name=plot_name.replace("-", "_"),
        figure_title=plot_name,
        subplot_layout={
            "ncols": 2,
            "nrows": 1,
            "sharex": "all",
            "sharey": "all",
        },
    )

    temp_stat = stock_price_statistic.StockPriceStatistic(quantile=quantile)
    temp_stat.set_statistics(time_series)
    real_stat = stock_price_statistic.StockPriceStatistic(quantile=quantile)
    real_stat.set_statistics(real_series)

    tail_upper = tail_statistics.UpperTailStatistic(temp_stat)
    tail_upper.set_statistics(None)
    tail_lower = tail_statistics.LowerTailStatistic(temp_stat)
    tail_lower.set_statistics(None)

    tail_upper_real = tail_statistics.UpperTailStatistic(temp_stat)
    tail_upper_real.set_statistics(None)
    tail_lower_real = tail_statistics.LowerTailStatistic(temp_stat)
    tail_lower_real.set_statistics(None)

    lower_comparator = distribution_comparator.DistributionComparator(
        real_stat=tail_lower_real,
        syn_stat=tail_lower,
        ax_style={
            "title": "Upper Tail Log Return",
            "ylabel": r"Density",
            "xlabel": r"log returns",
            "xscale": "linear",
            "yscale": "log",
        },
    )
    upper_comparator = distribution_comparator.DistributionComparator(
        real_stat=tail_upper_real,
        syn_stat=tail_upper,
        ax_style={
            "title": "Upper Tail Log Return",
            "ylabel": r"Density",
            "xlabel": r"log returns",
            "xscale": "linear",
            "yscale": "log",
        },
    )

    lower_comparator.draw_distributions(ax=plot.axes[0])
    upper_comparator.draw_distributions(ax=plot.axes[1])

    return plot
