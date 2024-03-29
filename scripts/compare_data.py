import click
import numpy as np
import real_data_loader
import garch_generator
import stock_price_statistic
import log_return_statistic
import scaled_log_return_statistic
import gen_data_loader
import scipy.stats as ss
import scipy.spatial.distance as sd
import tail_statistics
import distribution_comparator
import plotter


@click.group()
def inspect():
    pass

@inspect.command()
def compare_garch():

    data_loader = real_data_loader.RealDataLoader()
    real_stock_data = data_loader.get_timeseries(col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False)
    real_stock_data = real_stock_data.iloc[:,[1]]

    garch = garch_generator.GarchGenerator(name='GARCH_1_1_normal')
    gen_loader = gen_data_loader.GenDataLoader()
    synth_data = gen_loader.get_timeseries(garch, data_loader=data_loader, col_name='Adj Close')
    synth_data = synth_data.iloc[:,[1]]
    
    quantile = 0.01
    real_data = stock_price_statistic.StockPriceStatistic(quantile=quantile)
    real_data.set_statistics(real_stock_data)
    real_log = log_return_statistic.LogReturnStatistic(quantile=quantile)
    real_log.set_statistics(real_stock_data)
    real_scaled_log = scaled_log_return_statistic.ScaledLogReturnStatistic(quantile=quantile, window=50)
    real_scaled_log.set_statistics(real_stock_data)
    real_tail_upper = tail_statistics.UpperTailStatistic(real_scaled_log)
    real_tail_upper.set_statistics(None)
    real_tail_lower = tail_statistics.LowerTailStatistic(real_scaled_log)
    real_tail_lower.set_statistics(None)

    synthetic_data = stock_price_statistic.StockPriceStatistic(quantile=quantile, legend_postfix=" - GARCH")
    synthetic_data.set_statistics(synth_data)
    synthetic_log = log_return_statistic.LogReturnStatistic(quantile=quantile, legend_postfix=" - GARCH")
    synthetic_log.set_statistics(synth_data)
    syn_scaled_log = scaled_log_return_statistic.ScaledLogReturnStatistic(quantile=quantile, window=50)
    syn_scaled_log.set_statistics(synth_data)
    syn_tail_upper = tail_statistics.UpperTailStatistic(syn_scaled_log)
    syn_tail_upper.set_statistics(None)
    syn_tail_lower = tail_statistics.LowerTailStatistic(syn_scaled_log)
    syn_tail_lower.set_statistics(None)

    plot = plotter.Plotter(
        cache='data/cache',
        figure_name='distribution_divergence',
        figure_title='Distribution Divergences',
        subplot_layout=(2,2))

    tail_comp_lower = distribution_comparator.DistributionComparator(
        real_stat=real_tail_lower,
        syn_stat=syn_tail_lower,
        ax_style = {
            'title' : 'Lower Tail Statistic',
            'ylabel' : r'Density',
            'xlabel' : r'scaled log returns',
            'xscale' : 'linear',
            'yscale' : 'linear',
            }
    )
    tail_comp_lower.draw_distributions(plot.axes[0,0])

    tail_comp_upper = distribution_comparator.DistributionComparator(
        real_stat=real_tail_upper,
        syn_stat=syn_tail_upper,
        ax_style = {
            'title' : 'Upper Tail Statistic',
            'ylabel' : r'Density',
            'xlabel' : r'scaled log returns',
            'xscale' : 'linear',
            'yscale' : 'linear',
            }
    )
    tail_comp_upper.draw_distributions(plot.axes[0,1])

    log_comp = distribution_comparator.DistributionComparator(
        real_stat=real_log,
        syn_stat=synthetic_log,
        ax_style = {
            'title' : 'Log Return Statistic',
            'ylabel' : r'Density',
            'xlabel' : r'log returns',
            'xscale' : 'linear',
            'yscale' : 'linear',
            }
    )
    log_comp.draw_distributions(plot.axes[1,0])

    scaled_log_comp = distribution_comparator.DistributionComparator(
        real_stat=real_scaled_log,
        syn_stat=syn_scaled_log,
        ax_style = {
            'title' : 'Scaled Log Return Statistic',
            'ylabel' : r'Density',
            'xlabel' : r'scaled log returns',
            'xscale' : 'linear',
            'yscale' : 'linear',
            }
    )
    scaled_log_comp.draw_distributions(plot.axes[1,1])
    plot.save()
    
if __name__ == "__main__":
    compare_garch()