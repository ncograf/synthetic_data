import click
import numpy as np
import pandas as pd
import data_inspector
import real_data_loader
from pathlib import Path
from typing import Optional, Set, List, Literal

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
import gen_data_loader
import time
import plotter


@click.group()
def inspect():
    pass

#@inspect.command()
def visualize_stylized_pair(fact : Literal['heavy-tails', 'lin-upred', 'vol-culster', 'lev-effect', 'coarse-fine-vol', 'gain-loss-asym']):

    real_loader = real_data_loader.RealDataLoader()
    garch = garch_generator.GarchGenerator(p=1, q=1, name='GARCH_1_1_normal')
    data_loader = gen_data_loader.GenDataLoader()
    gen_data = data_loader.get_timeseries(generator=garch, col_name="Adj Close", data_loader=real_loader,  update_all=False)
    real_data = real_loader.get_timeseries(col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False)
    
    visualize_pair(real_data=real_data, gen_data = gen_data, fact=fact)

#@inspect.command()
def visualize_stylized_facts(loader : Literal['real', 'g_1_1_norm']):
    real_loader = real_data_loader.RealDataLoader()
    if loader == 'g_1_1_norm':
        garch = garch_generator.GarchGenerator(p=1, q=1, name='GARCH_1_1_normal')
        data_loader = gen_data_loader.GenDataLoader()
        stock_data = data_loader.get_timeseries(generator=garch, col_name="Adj Close", data_loader=real_loader,  update_all=False)
    elif loader == 'real':
        stock_data = real_loader.get_timeseries(col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False)
    else:
        raise ValueError(f"The given loader {loader} is not supported.")
    
    visualize_all(stock_data=stock_data)

def visualize_pair(
    real_data : pd.DataFrame, gen_data : pd.DataFrame,
    fact : Literal['heavy-tails', 'lin-upred', 'vol-culster', 'lev-effect', 'coarse-fine-vol', 'gain-loss-asym']
    ):

    plot = plotter.Plotter(
        cache='data/cache',
        figure_name=fact.replace('-', '_'),
        figure_title=fact,
        subplot_layout=(1,2))

    m_lag = 1000
    
    if fact == 'heavy-tails':

        for i, data in enumerate((real_data, gen_data)):
            log_stat = log_return_statistic.LogReturnStatistic(0.001)
            log_stat.set_statistics(data)

            norm_price_ret = normalized_price_return.NormalizedPriceReturn(log_stat)
            norm_price_ret.set_statistics(None)
            norm_price_ret.get_alphas()
            norm_price_ret.draw_stylized_fact(plot.axes[i])
    
    elif fact == 'lin-upred':

        for i, (data, pref) in enumerate(zip([real_data, gen_data], ['real', 'generated'])):
            log_stat = log_return_statistic.LogReturnStatistic(0.001)
            log_stat.set_statistics(data)

            log_corr_stat = auto_corr_statistic.AutoCorrStatistic(max_lag=m_lag, underlaying=log_stat, title='linear unpredictability ' + pref, xscale='log', yscale='linear', ylim=(-1,1), implementation='boosted')
            log_corr_stat.set_statistics(None)
            log_corr_stat.draw_stylized_fact(plot.axes[i])

    elif fact == 'vol-cluster':

        for i, (data, pref) in enumerate(zip([real_data, gen_data], ['real', 'generated'])):
            abs_log_stat = abs_log_return_statistic.AbsLogReturnStatistic(0.001)
            abs_log_stat.set_statistics(data)

            volatility_clustering = auto_corr_statistic.AutoCorrStatistic(max_lag=m_lag, underlaying=abs_log_stat, title='volatility clustering ' + pref, xscale='log', yscale='log', ylim=None, powerlaw=True, implementation='boosted')
            volatility_clustering.set_statistics(None)
            volatility_clustering.draw_stylized_fact(plot.axes[i])

    elif fact == 'lev-effect':

        for i, (data, pref) in enumerate(zip([real_data, gen_data], ['real', 'generated'])):
    
            log_stat = log_return_statistic.LogReturnStatistic(0.001)
            log_stat.set_statistics(data)

            lev_eff = leverage_effect.LeverageEffect(max_lag=100, underlaying=log_stat)
            lev_eff.set_statistics(None)
            lev_eff.draw_stylized_fact(plot.axes[i])

    elif fact == 'coarse-fine-vol':

        for i, (data, pref) in enumerate(zip([real_data, gen_data], ['real', 'generated'])):

            log_stat = log_return_statistic.LogReturnStatistic(0.001)
            log_stat.set_statistics(data)
    
            co_fine = coarse_fine_volatility.CoarseFineVolatility(max_lag=25, tau=5, underlaying=log_stat)
            co_fine.set_statistics(None)
            co_fine.draw_stylized_fact(plot.axes[i])

    elif fact == 'gain-loss-asym':

        for i, (data, pref) in enumerate(zip([real_data, gen_data], ['real', 'generated'])):

            price = stock_price_statistic.StockPriceStatistic(0.001)
            price.set_statistics(data)
    
            gain_loss_asym = gain_loss_asymetry.GainLossAsymetry(max_lag=m_lag, theta=0.1, underlying_price=price)
            gain_loss_asym.set_statistics(None)
            gain_loss_asym.draw_stylized_fact(plot.axes[i])

    plot.save()


def visualize_all(stock_data : pd.DataFrame):
    
    log_stat = log_return_statistic.LogReturnStatistic(0.001)
    log_stat.set_statistics(stock_data)

    ret_stat = return_statistic.ReturnStatistic(0.001)
    ret_stat.set_statistics(stock_data)

    abs_log_stat = abs_log_return_statistic.AbsLogReturnStatistic(0.001)
    abs_log_stat.set_statistics(stock_data)

    price = stock_price_statistic.StockPriceStatistic(0.001)
    price.set_statistics(stock_data)

    m_lag = 1000
    log_corr_stat = auto_corr_statistic.AutoCorrStatistic(max_lag=m_lag, underlaying=log_stat, title='linear unpredictability', xscale='log', yscale='linear', ylim=(-1,1), implementation='boosted')
    log_corr_stat.set_statistics(None)

    volatility_clustering = auto_corr_statistic.AutoCorrStatistic(max_lag=m_lag, underlaying=abs_log_stat, title='volatility clustering', xscale='log', yscale='log', ylim=None, powerlaw=True, implementation='boosted')
    volatility_clustering.set_statistics(None)

    norm_price_ret = normalized_price_return.NormalizedPriceReturn(log_stat)
    norm_price_ret.set_statistics(None)
    norm_price_ret.get_alphas()
    
    lev_eff = leverage_effect.LeverageEffect(max_lag=100, underlaying=ret_stat)
    lev_eff.set_statistics(None)

    co_fine = coarse_fine_volatility.CoarseFineVolatility(max_lag=25, tau=5, underlaying=log_stat)
    co_fine.set_statistics(None)

    gain_loss_asym = gain_loss_asymetry.GainLossAsymetry(max_lag=m_lag, theta=0.1, underlying_price=price)
    gain_loss_asym.set_statistics(None)

    plot = plotter.Plotter(
        cache='data/cache',
        figure_name='stylized_facts',
        figure_title='stylized_facts',
        subplot_layout=(2,3))

    log_corr_stat.draw_stylized_fact(plot.axes[0,0])
    norm_price_ret.draw_stylized_fact(plot.axes[0,1])
    volatility_clustering.draw_stylized_fact(plot.axes[0,2])
    lev_eff.draw_stylized_fact(plot.axes[1,0])
    co_fine.draw_stylized_fact(plot.axes[1,1])
    gain_loss_asym.draw_stylized_fact(plot.axes[1,2])

    plot.save()

if __name__ == "__main__":

    #visualize_stylized_facts(loader='g_1_1_norm')
    #visualize_stylized_pair(fact='lin-upred')
    #visualize_stylized_pair(fact='coarse-fine-vol')
    #visualize_stylized_pair(fact='gain-loss-asym')
    #visualize_stylized_pair(fact='lev-effect')
    visualize_stylized_pair(fact='vol-cluster')
    #visualize_stylized_pair(fact='heavy-tails')