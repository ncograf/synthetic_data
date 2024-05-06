from typing import Literal

import abs_log_return_statistic
import auto_corr_statistic
import coarse_fine_volatility
import gain_loss_asymetry
import leverage_effect
import log_return_statistic
import normalized_price_return
import pandas as pd
import plotter
import return_statistic
import stock_price_statistic


def visualize_pair(
    real_data: pd.DataFrame,
    gen_data: pd.DataFrame,
    fact: Literal[
        "heavy-tails",
        "lin-upred",
        "vol-culster",
        "lev-effect",
        "coarse-fine-vol",
        "gain-loss-asym",
    ],
) -> plotter.Plotter:
    """Plot a stylized fact for two datasets.

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
        figure_name=fact.replace("-", "_"),
        figure_title=fact,
        subplot_layout={
            "ncols": 2,
            "nrows": 1,
            "sharex": "all",
            "sharey": "all",
        },
    )

    m_lag = 1000

    if fact == "heavy-tails":
        for i, (data, pref) in enumerate(
            zip([real_data, gen_data], [" real data", " generated data"])
        ):
            log_stat = log_return_statistic.LogReturnStatistic(0.001)
            log_stat.set_statistics(data)

            norm_price_ret = normalized_price_return.NormalizedPriceReturn(
                log_stat, title_postfix=pref
            )
            norm_price_ret.set_statistics(None)
            norm_price_ret.get_alphas()
            norm_price_ret.draw_stylized_fact(plot.axes[i])

    elif fact == "lin-upred":
        for i, (data, pref) in enumerate(
            zip([real_data, gen_data], ["real", "generated"])
        ):
            log_stat = log_return_statistic.LogReturnStatistic(0.001)
            log_stat.set_statistics(data)

            log_corr_stat = auto_corr_statistic.AutoCorrStatistic(
                max_lag=m_lag,
                underlaying=log_stat,
                title="linear unpredictability " + pref,
                xscale="log",
                yscale="linear",
                ylim=(-1, 1),
                implementation="boosted",
            )
            log_corr_stat.set_statistics(None)
            log_corr_stat.draw_stylized_fact(plot.axes[i])

    elif fact == "vol-cluster":
        for i, (data, pref) in enumerate(
            zip([real_data, gen_data], ["real", "generated"])
        ):
            abs_log_stat = abs_log_return_statistic.AbsLogReturnStatistic(0.001)
            abs_log_stat.set_statistics(data)

            volatility_clustering = auto_corr_statistic.AutoCorrStatistic(
                max_lag=m_lag,
                underlaying=abs_log_stat,
                title="volatility clustering " + pref,
                xscale="log",
                yscale="log",
                ylim=None,
                powerlaw=True,
                implementation="boosted",
            )
            volatility_clustering.set_statistics(None)
            volatility_clustering.draw_stylized_fact(plot.axes[i])

    elif fact == "lev-effect":
        for i, (data, pref) in enumerate(
            zip([real_data, gen_data], [" real data", " generated data"])
        ):
            log_stat = log_return_statistic.LogReturnStatistic(0.001)
            log_stat.set_statistics(data)

            lev_eff = leverage_effect.LeverageEffect(
                max_lag=100, underlaying=log_stat, title_postfix=pref
            )
            lev_eff.set_statistics(None)
            lev_eff.draw_stylized_fact(plot.axes[i])

    elif fact == "coarse-fine-vol":
        for i, (data, pref) in enumerate(
            zip([real_data, gen_data], [" real data", " generated data"])
        ):
            log_stat = log_return_statistic.LogReturnStatistic(0.001)
            log_stat.set_statistics(data)

            co_fine = coarse_fine_volatility.CoarseFineVolatility(
                max_lag=25, tau=5, underlaying=log_stat, title_postfix=pref
            )
            co_fine.set_statistics(None)
            co_fine.draw_stylized_fact(plot.axes[i])

    elif fact == "gain-loss-asym":
        for i, (data, pref) in enumerate(
            zip([real_data, gen_data], [" real data", " generated data"])
        ):
            price = stock_price_statistic.StockPriceStatistic()
            price.set_statistics(data)

            gain_loss_asym = gain_loss_asymetry.GainLossAsymetry(
                max_lag=m_lag, theta=0.1, underlying_price=price, title_postfix=pref
            )
            gain_loss_asym.set_statistics(None)
            gain_loss_asym.draw_stylized_fact(plot.axes[i])

    return plot


def visualize_all(stock_data: pd.DataFrame, name: str = "") -> plotter.Plotter:
    """Visualizes all stilized facts and returns the plotter object

    Args:
        stock_data (pd.DataFrame): _description_
        name (str, optional): _description_. Defaults to "".

    Returns:
        Plotter: Python Plotter object containing the plots
    """

    log_stat = log_return_statistic.LogReturnStatistic()
    log_stat.set_statistics(stock_data)

    ret_stat = return_statistic.ReturnStatistic()
    ret_stat.set_statistics(stock_data)

    abs_log_stat = abs_log_return_statistic.AbsLogReturnStatistic()
    abs_log_stat.set_statistics(stock_data)

    price = stock_price_statistic.StockPriceStatistic()
    price.set_statistics(stock_data)

    m_lag = 1000

    if m_lag >= ret_stat.statistic.shape[0]:
        raise ValueError(
            f"Trying stylized facts with lag {m_lag}, log statistic only has {ret_stat.shape[0]} elements."
        )

    log_corr_stat = auto_corr_statistic.AutoCorrStatistic(
        max_lag=m_lag,
        underlaying=log_stat,
        title="linear unpredictability",
        xscale="log",
        yscale="linear",
        ylim=(-1, 1),
        implementation="boosted",
    )
    log_corr_stat.set_statistics(None)

    volatility_clustering = auto_corr_statistic.AutoCorrStatistic(
        max_lag=m_lag,
        underlaying=abs_log_stat,
        title="volatility clustering",
        xscale="log",
        yscale="log",
        ylim=None,
        powerlaw=True,
        implementation="boosted",
    )
    volatility_clustering.set_statistics(None)

    norm_price_ret = normalized_price_return.NormalizedPriceReturn(log_stat)
    norm_price_ret.set_statistics(None)
    norm_price_ret.get_alphas()

    lev_eff = leverage_effect.LeverageEffect(max_lag=100, underlaying=log_stat)
    lev_eff.set_statistics(None)

    co_fine = coarse_fine_volatility.CoarseFineVolatility(
        max_lag=25, tau=5, underlaying=log_stat
    )
    co_fine.set_statistics(None)

    gain_loss_asym = gain_loss_asymetry.GainLossAsymetry(
        max_lag=m_lag, theta=0.1, underlying_price=price
    )
    gain_loss_asym.set_statistics(None)

    plot = plotter.Plotter(
        cache="data/cache",
        figure_name="stylized_facts_" + name,
        figure_title="Stylized Facts " + name,
        figure_style={
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
        },
        subplot_layout={
            "ncols": 3,
            "nrows": 2,
            "sharex": "none",
            "sharey": "none",
        },
    )

    log_corr_stat.draw_stylized_fact(plot.axes[0, 0])
    norm_price_ret.draw_stylized_fact(plot.axes[0, 1])
    volatility_clustering.draw_stylized_fact(plot.axes[0, 2])
    lev_eff.draw_stylized_fact(plot.axes[1, 0])
    co_fine.draw_stylized_fact(plot.axes[1, 1])
    gain_loss_asym.draw_stylized_fact(plot.axes[1, 2])

    return plot
